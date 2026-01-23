#[cfg(test)]
mod test;

use alloc::fmt;
use core::iter::Peekable;
use std::borrow::Cow;

use logos::Span;

use crate::{
    ast::{
        Args, AttName, BinOp, Block, Exp, Field, FuncBody, FunctionCall, PrefixExp, SYNTHETIC_POS,
        SYNTHETIC_SPAN, Spanned, Stat, UnOp, Var,
    },
    lex::Token,
    s, synth,
};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum TokenClass {
    And,
    Assign,
    Comma,
    DotDot,
    Equals,
    Not,
    Or,
    Plus,
    Slash,
    Semi,

    Ident,
    Number,
    StringLiteral,

    OParen,
    CParen,
    OBrace,
    CBrace,
    OSquare,
    CSquare,
    LAngle,
    RAngle,

    Then,

    Function,
    Expression,
    EndOfBlock,
    Eof,
    Field,
    Statement,
    Var,
}

impl From<Token<'_>> for TokenClass {
    fn from(value: Token<'_>) -> Self {
        match value {
            Token::OParen => TokenClass::OParen,
            Token::CParen => TokenClass::CParen,
            Token::OBrace => TokenClass::OBrace,
            Token::CBrace => TokenClass::CBrace,
            Token::OSquare => TokenClass::OSquare,
            Token::CSquare => TokenClass::CSquare,
            Token::Comma => TokenClass::Comma,
            x => todo!("{x:?}"),
        }
    }
}

#[derive(Debug, PartialEq)] // TODO: smarter than this
pub enum ParseError<'a, E> {
    ExpectedGot {
        span: Span,
        expected: Vec<TokenClass>,
        got: Option<Token<'a>>,
    },
    Lexer(E),
}

pub type ParseResult<'a, T, E> = Result<Spanned<T>, (Spanned<T>, Vec<ParseError<'a, E>>)>;

fn unbox<'a, T, E>(
    result: ParseResult<'a, T, E>,
    result_span: &mut Option<Span>,
    errors: &mut Vec<ParseError<'a, E>>,
) -> Spanned<T> {
    update_span_from_parse_result(result_span, &result);
    match result {
        Ok(x) => x,
        Err((x, mut e)) => {
            errors.append(&mut e);
            x
        }
    }
}

fn update_span(result_span: &mut Option<Span>, span: &Span) {
    if let Some(result_span) = result_span.as_mut() {
        if span.end != SYNTHETIC_POS {
            // Don't extend if the new end pos is synthetic
            result_span.end = result_span.end.max(span.end); // only ever extend the span
        }
    } else {
        *result_span = Some(span.clone());
    }
}

fn update_span_from_parse_result<'a, T, E>(
    result_span: &mut Option<Span>,
    parse_result: &ParseResult<'a, T, E>,
) {
    match parse_result {
        Ok(Spanned(_, span)) | Err((Spanned(_, span), _)) => update_span(result_span, span),
    }
}

fn update_span_from_peek_result<'a, E>(
    result_span: &mut Option<Span>,
    peek_result: Option<&(Result<Token<'a>, E>, Span)>,
) {
    if let Some((_, span)) = peek_result {
        update_span(result_span, span);
    }
}

fn lex_error<'a, E: Clone + fmt::Debug>(
    error: &E,
    span: &Span,
    result_span: &mut Option<Span>,
    errors: &mut Vec<ParseError<'a, E>>,
) {
    update_span(result_span, span);
    errors.push(ParseError::Lexer(error.clone()));
}

fn expected_got_error_from_peek_result<'a, E>(
    expected: Vec<TokenClass>,
    got: Option<&(Result<Token<'a>, E>, Span)>,
    result_span: &mut Option<Span>,
    errors: &mut Vec<ParseError<'a, E>>,
) {
    update_span_from_peek_result(result_span, got);
    errors.push(ParseError::ExpectedGot {
        span: result_span.clone().unwrap_or(SYNTHETIC_SPAN),
        expected,
        got: got.and_then(|(x, _)| x.as_ref().ok().cloned()),
    });
}

fn recover<'a, E>(
    input: &mut Peekable<impl Iterator<Item = (Result<Token<'a>, E>, Span)>>,
    recovery_points: Vec<Token<'a>>,
) {
    loop {
        match input.peek() {
            Some((Ok(x), _)) => {
                if recovery_points.contains(x) {
                    break;
                }
                input.next();
            }
            None => break, // Welp. EOF. Callee can trip on that and make a new error.
            _ => {
                input.next();
            }
        }
    }
}

fn finish<'a, T, E>(val: Spanned<T>, errors: Vec<ParseError<'a, E>>) -> ParseResult<'a, T, E> {
    if errors.is_empty() {
        Ok(val)
    } else {
        Err((val, errors))
    }
}

// TODO: resultify
pub fn parse_string(x: &str) -> Box<[u8]> {
    let mut iter = x.bytes();

    let start_char = iter.next().unwrap();
    assert!(start_char == b'"' || start_char == b'\'');

    let mut result = Vec::new();
    while let Some(x) = iter.next() {
        if x == start_char {
            break;
        } else if x == b'\\' {
            todo!()
        } else {
            result.push(x);
        }
    }
    result.into_boxed_slice()
}

pub type ParseFn<'a, E: Clone + fmt::Debug, T, I: Iterator<Item = (Result<Token<'a>, E>, Span)>> =
    fn(&mut Peekable<I>) -> ParseResult<'a, T, E>;

#[inline]
pub fn parse_list<
    'a,
    E: Clone + fmt::Debug,
    T,
    I: Iterator<Item = (Result<Token<'a>, E>, Span)>,
>(
    input: &mut Peekable<I>,
    start: Token<'a>,
    end: Token<'a>,
    delimiter: Token<'a>,
    allow_trailing: bool,
    item_class: TokenClass,
    item_parser: ParseFn<'a, E, T, I>,
) -> ParseResult<'a, Vec<Spanned<T>>, E> {
    let mut result_span = None;
    let mut errors = vec![];
    let mut list = vec![];

    match input.next() {
        Some((Err(e), span)) => {
            // Welp. How did we get here anyway? Calling function should've checked this.
            // Still not crashing because I may break that contract.
            lex_error(&e, &span, &mut result_span, &mut errors);
        }
        Some((Ok(x), span)) if x == start => {
            // Yay.
            update_span(&mut result_span, &span);
        }
        x => {
            // Same as above; should've been checked, but not crashing in case I changed my mind about that guarantee.
            expected_got_error_from_peek_result(
                vec![start.into()],
                x.as_ref(),
                &mut result_span,
                &mut errors,
            );
        }
    }

    let mut allow_close = true;
    let mut allow_item = true;
    let mut allow_delimiter = false;
    loop {
        // Figure out what the error should be if we get one
        let mut expected = Vec::new();
        if allow_close {
            expected.push(end.clone().into());
        }
        if allow_item {
            expected.push(item_class);
        }
        if allow_delimiter {
            expected.push(delimiter.clone().into());
        }
        match input.peek() {
            Some((Err(e), span)) => {
                lex_error(e, span, &mut result_span, &mut errors);
                input.next();
            }
            t @ Some((Ok(x), span)) if x == &delimiter => {
                if !allow_delimiter {
                    // Oops.
                    expected_got_error_from_peek_result(expected, t, &mut result_span, &mut errors);
                }
                // Regardless, update state
                update_span(&mut result_span, span);
                input.next();
                allow_close = allow_trailing; // only allow the end of the list if trailing delimiters are allowed
                allow_item = true; // always allow an iterm after a delimiter
                allow_delimiter = false; // never allow two delimiters in a row
            }
            t @ Some((Ok(x), span)) if x == &end => {
                if !allow_close {
                    // Oops.
                    expected_got_error_from_peek_result(expected, t, &mut result_span, &mut errors);
                }
                // Regardless, call the list done. The user clearly intended to close it.
                update_span(&mut result_span, span);
                input.next();
                break;
            }
            x => {
                // Assume this is supposed to start an item. Or it's an error anyway.
                if !allow_item {
                    // Oops.
                    expected_got_error_from_peek_result(expected, x, &mut result_span, &mut errors);
                }
                list.push(unbox(item_parser(input), &mut result_span, &mut errors));
                allow_close = true; // always allow the list to end directly after an item
                allow_item = false; // never allow two items without a delimiter
                allow_delimiter = true; // always allow a delimiter after a list item
            }
        }
    }

    finish(s!(list, result_span.unwrap_or(SYNTHETIC_SPAN)), errors)
}

fn parse_ident<'a, E: Clone + fmt::Debug>(
    input: &mut Peekable<impl Iterator<Item = (Result<Token<'a>, E>, Span)>>,
) -> ParseResult<'a, Cow<'a, str>, E> {
    let mut result_span = None;
    let mut errors = vec![];

    let name = match input.next() {
        Some((Err(e), span)) => {
            lex_error(&e, &span, &mut result_span, &mut errors);
            "<error>".into()
        }
        Some((Ok(Token::Ident(x)), span)) => {
            update_span(&mut result_span, &span);
            x.into()
        }
        x => {
            expected_got_error_from_peek_result(
                vec![TokenClass::Ident],
                x.as_ref(),
                &mut result_span,
                &mut errors,
            );
            "<error>".into()
        }
    };

    finish(s!(name, result_span.unwrap_or(SYNTHETIC_SPAN)), errors)
}

fn parse_table_field<'a, E: Clone + fmt::Debug>(
    input: &mut Peekable<impl Iterator<Item = (Result<Token<'a>, E>, Span)>>,
) -> ParseResult<'a, Field<'a>, E> {
    todo!()
}

fn parse_atom<'a, E: Clone + fmt::Debug>(
    input: &mut Peekable<impl Iterator<Item = (Result<Token<'a>, E>, Span)>>,
) -> ParseResult<'a, Exp<'a>, E> {
    let mut result_span = None;
    let mut errors = vec![];

    let result_atom = match input.peek() {
        Some((Ok(Token::Nil), span)) => {
            update_span(&mut result_span, span);
            input.next();
            Some(Exp::Nil)
        }
        Some((Ok(Token::True), span)) => {
            update_span(&mut result_span, span);
            input.next();
            Some(Exp::True)
        }
        Some((Ok(Token::False), span)) => {
            update_span(&mut result_span, span);
            input.next();
            Some(Exp::False)
        }
        Some((Ok(Token::Number(x)), span)) => {
            update_span(&mut result_span, span);
            let str = *x;
            let span = span.clone();
            input.next();
            if let Ok(x) = str.parse() {
                Some(Exp::NumeralInt(s!(x, span.clone())))
            } else if let Ok(x) = str.parse() {
                Some(Exp::NumeralFloat(s!(x, span.clone())))
            } else {
                todo!("more interesting numerals")
            }
        }
        Some((Ok(Token::StringLiteral(x)), span)) => {
            update_span(&mut result_span, span);
            let str = parse_string(x);
            let span = span.clone();
            input.next();
            Some(Exp::LiteralString(s!(str, span)))
        }
        Some((Ok(Token::OBrace), _)) => {
            let list = parse_list(
                input,
                Token::OBrace,
                Token::CBrace,
                Token::Comma,
                true,
                TokenClass::Field,
                parse_table_field,
            );
            Some(Exp::TableConstructor(unbox(
                list,
                &mut result_span,
                &mut errors,
            )))
        }
        Some((Ok(Token::Ident(_) | Token::OParen), _)) => Some(Exp::PrefixExp(unbox(
            parse_prefix_exp(input),
            &mut result_span,
            &mut errors,
        ))),
        Some((Ok(Token::Function), span)) => {
            update_span(&mut result_span, span);
            input.next();
            Some(Exp::FunctionDef(unbox(
                parse_func_body(input),
                &mut result_span,
                &mut errors,
            )))
        }
        Some((Ok(op @ (Token::Minus | Token::Not | Token::Count | Token::BitNot)), span)) => {
            let op = match op {
                Token::Minus => UnOp::Unm,
                Token::Not => UnOp::Not,
                Token::Count => UnOp::Len,
                Token::BitNot => UnOp::Bnot,
                _ => unreachable!(),
            };
            update_span(&mut result_span, span);
            input.next();
            Some(Exp::UnExp {
                op,
                rhs: Box::new(unbox(
                    parse_exp_inner(input, 20 /* unary op precedence */),
                    &mut result_span,
                    &mut errors,
                )),
            })
        }
        x => {
            expected_got_error_from_peek_result(
                vec![TokenClass::Expression],
                x,
                &mut result_span,
                &mut errors,
            );
            None
        }
    }
    .unwrap_or_else(|| {
        recover(
            input,
            vec![
                // TODO: Recover at operators so parse_exp_inner can continue parsing even if an individual atom is a problem.
                Token::CParen,
                Token::CBrace,
                Token::CSquare,
                Token::Comma,
                Token::Assign,
            ],
        );
        Exp::Error
    });

    finish(
        s!(result_atom, result_span.unwrap_or(SYNTHETIC_SPAN)),
        errors,
    )
}

// Pratt parsing!
// Reference: https://matklad.github.io/2020/04/13/simple-but-powerful-pratt-parsing.html
pub fn parse_exp_inner<'a, E: Clone + fmt::Debug>(
    input: &mut Peekable<impl Iterator<Item = (Result<Token<'a>, E>, Span)>>,
    min_bp: u8,
) -> ParseResult<'a, Exp<'a>, E> {
    let mut result_span = None;
    let mut errors = vec![];

    let mut lhs = unbox(parse_atom(input), &mut result_span, &mut errors).0; // We're tracking span separately; extract the value

    loop {
        let old_span = result_span.clone();
        let op = match input.peek() {
            Some((Ok(Token::Plus), _)) => BinOp::Add,
            Some((Ok(Token::Minus), _)) => BinOp::Sub,
            Some((Ok(Token::Star), _)) => BinOp::Mul,
            Some((Ok(Token::Slash), _)) => BinOp::Div,
            Some((Ok(Token::IntDiv), _)) => BinOp::Idiv,
            Some((Ok(Token::Hat), _)) => BinOp::Pow,
            Some((Ok(Token::Modulo), _)) => BinOp::Mod,
            Some((Ok(Token::BitAnd), _)) => BinOp::Band,
            Some((Ok(Token::BitNot), _)) => BinOp::Bxor,
            Some((Ok(Token::BitOr), _)) => BinOp::Bor,
            Some((Ok(Token::RightShift), _)) => BinOp::Shr,
            Some((Ok(Token::LeftShift), _)) => BinOp::Shl,
            Some((Ok(Token::DotDot), _)) => BinOp::Concat,
            Some((Ok(Token::LeftAngle), _)) => BinOp::Lt,
            Some((Ok(Token::LessEquals), _)) => BinOp::Le,
            Some((Ok(Token::RightAngle), _)) => BinOp::Gt,
            Some((Ok(Token::GreaterEquals), _)) => BinOp::Ge,
            Some((Ok(Token::Equals), _)) => BinOp::Eq,
            Some((Ok(Token::NotEquals), _)) => BinOp::Neq,
            Some((Ok(Token::And), _)) => BinOp::And,
            Some((Ok(Token::Or), _)) => BinOp::Or,
            _ => break, // no longer an expression; bail
        };

        let (l_bp, r_bp) = op.binding_power();
        if l_bp < min_bp {
            break;
        }

        input.next();

        let rhs = unbox(parse_exp_inner(input, r_bp), &mut result_span, &mut errors);

        lhs = Exp::BinExp {
            lhs: Box::new(s!(lhs, old_span.unwrap_or(SYNTHETIC_SPAN))),
            op,
            rhs: Box::new(rhs),
        };
    }

    finish(s!(lhs, result_span.unwrap_or(SYNTHETIC_SPAN)), errors)
}

pub fn parse_exp<'a, E: Clone + fmt::Debug>(
    input: &mut Peekable<impl Iterator<Item = (Result<Token<'a>, E>, Span)>>,
) -> ParseResult<'a, Exp<'a>, E> {
    parse_exp_inner(input, 0)
}

pub fn parse_prefix_exp<'a, E: Clone + fmt::Debug>(
    input: &mut Peekable<impl Iterator<Item = (Result<Token<'a>, E>, Span)>>,
) -> ParseResult<'a, PrefixExp<'a>, E> {
    let mut result_span = None;
    let mut errors = vec![];

    let mut lhs = match input.peek() {
        Some((Err(e), span)) => {
            lex_error(e, span, &mut result_span, &mut errors);
            None
        }
        Some((Ok(Token::OParen), _)) => {
            input.next();
            let inner = unbox(parse_exp(input), &mut result_span, &mut errors);
            match input.next() {
                Some((Err(e), span)) => {
                    // welp.
                    lex_error(&e, &span, &mut result_span, &mut errors);
                }
                Some((Ok(Token::CParen), span)) => {
                    update_span(&mut result_span, &span);
                }
                x => {
                    expected_got_error_from_peek_result(
                        vec![TokenClass::CParen],
                        x.as_ref(),
                        &mut result_span,
                        &mut errors,
                    );
                }
            }
            Some(PrefixExp::Parens(Box::new(inner)))
        }
        Some((Ok(Token::Ident(x)), span)) => {
            update_span(&mut result_span, span);
            let str = (*x).into();
            let span = span.clone();
            input.next();
            Some(PrefixExp::Var(s!(Var::Name(s!(str, span.clone())), span)))
        }
        x => {
            expected_got_error_from_peek_result(
                vec![TokenClass::OParen, TokenClass::Ident],
                x,
                &mut result_span,
                &mut errors,
            );
            None
        }
    }
    .unwrap_or_else(|| {
        recover(
            input,
            vec![
                Token::If,
                Token::Else,
                Token::ElseIf,
                Token::Semi,
                Token::End,
            ],
        );
        PrefixExp::Error
    });

    loop {
        match input.peek() {
            Some((Err(e), span)) => {
                lex_error(e, span, &mut result_span, &mut errors);
                input.next(); // skip past the lexer error
                break; // before breaking because we probably are supposed to be done with this expression
            }
            Some((
                Ok(x @ (Token::OParen | Token::Colon | Token::OBrace | Token::StringLiteral(_))),
                _,
            )) => {
                // Function/method call. Collected into one place for sanity and some measure of cleanliness.
                let old_span = result_span.clone().unwrap_or(SYNTHETIC_SPAN); // should always have something if we got here, but just in case
                let method = (x == &Token::Colon).then(|| {
                    input.next(); // Skip the colon
                    match input.next() {
                        Some((Err(e), span)) => {
                            lex_error(&e, &span, &mut result_span, &mut errors);
                            synth!("<error>".into())
                        }
                        Some((Ok(Token::Ident(x)), span)) => {
                            s!(x.into(), span)
                        }
                        x => {
                            expected_got_error_from_peek_result(
                                vec![TokenClass::Ident],
                                x.as_ref(),
                                &mut result_span,
                                &mut errors,
                            );
                            synth!("<error>".into())
                        }
                    }
                });
                let args = match input.peek() {
                    Some((Err(e), span)) => {
                        lex_error(e, &span, &mut result_span, &mut errors);
                        synth!(Args::Error)
                    }
                    Some((Ok(Token::StringLiteral(_)), _)) => todo!(),
                    Some((Ok(Token::OBrace), _)) => todo!(),
                    Some((Ok(Token::OParen), _)) => {
                        // expression list
                        let args = unbox(
                            parse_list(
                                input,
                                Token::OParen,
                                Token::CParen,
                                Token::Comma,
                                false,
                                TokenClass::Expression,
                                parse_exp,
                            ),
                            &mut result_span,
                            &mut errors,
                        );
                        let span = args.1.clone();
                        s!(Args::List(args), span)
                    }
                    x => {
                        expected_got_error_from_peek_result(
                            vec![
                                TokenClass::OParen,
                                TokenClass::OBrace,
                                TokenClass::StringLiteral,
                            ],
                            x,
                            &mut result_span,
                            &mut errors,
                        );
                        synth!(Args::Error)
                    }
                };
                lhs = PrefixExp::FunctionCall(s!(
                    FunctionCall {
                        lhs: Box::new(s!(lhs, old_span)),
                        method,
                        args,
                    },
                    result_span.clone().unwrap_or(SYNTHETIC_SPAN)
                ));
            }
            Some((Ok(Token::Dot), _)) => {
                // Path-like indexing
                let old_span = result_span.clone().unwrap_or(SYNTHETIC_SPAN);
                input.next();
                let member = match input.next() {
                    Some((Err(e), span)) => {
                        // welp.
                        lex_error(&e, &span, &mut result_span, &mut errors);
                        synth!("<error>".into())
                    }
                    Some((Ok(Token::Ident(x)), span)) => {
                        update_span(&mut result_span, &span);
                        s!(x.into(), span)
                    }
                    x => {
                        expected_got_error_from_peek_result(
                            vec![TokenClass::CSquare],
                            x.as_ref(),
                            &mut result_span,
                            &mut errors,
                        );
                        synth!("<error>".into())
                    }
                };
                lhs = PrefixExp::Var(s!(
                    Var::Path {
                        lhs: Box::new(s!(lhs, old_span)),
                        member
                    },
                    result_span.clone().unwrap_or(SYNTHETIC_SPAN)
                ));
            }
            Some((Ok(Token::OSquare), _)) => {
                // Array-like indexing
                let old_span = result_span.clone().unwrap_or(SYNTHETIC_SPAN); // should always have something if we got here, but just in case
                input.next();
                let idx = unbox(parse_exp(input), &mut result_span, &mut errors);
                match input.next() {
                    Some((Err(e), span)) => {
                        // welp.
                        lex_error(&e, &span, &mut result_span, &mut errors);
                    }
                    Some((Ok(Token::CSquare), span)) => {
                        update_span(&mut result_span, &span);
                    }
                    x => {
                        expected_got_error_from_peek_result(
                            vec![TokenClass::CSquare],
                            x.as_ref(),
                            &mut result_span,
                            &mut errors,
                        );
                    }
                }
                lhs = PrefixExp::Var(s!(
                    Var::Index {
                        lhs: Box::new(s!(lhs, old_span)),
                        idx: Box::new(idx),
                    },
                    result_span.clone().unwrap_or(SYNTHETIC_SPAN)
                ));
            }
            _ => {
                break; // not part of a prefixexp anymore; get out
            }
        }
    }
    finish(s!(lhs, result_span.unwrap_or(SYNTHETIC_SPAN)), errors)
}

pub fn parse_att_name<'a, E: Clone + fmt::Debug>(
    input: &mut Peekable<impl Iterator<Item = (Result<Token<'a>, E>, Span)>>,
) -> ParseResult<'a, AttName<'a>, E> {
    let mut result_span = None;
    let mut errors = vec![];

    let name = unbox(parse_ident(input), &mut result_span, &mut errors);

    let attrib = {
        match input.peek() {
            Some((Err(e), span)) => {
                lex_error(e, span, &mut result_span, &mut errors);
                input.next();
                None
            }
            Some((Ok(Token::LeftAngle), _)) => {
                input.next(); // Definitely an attribute; keep going.
                match input.next() {
                    Some((Err(e), span)) => {
                        lex_error(&e, &span, &mut result_span, &mut errors);
                        None
                    }
                    Some((Ok(Token::Ident(x)), str_span)) => match input.next() {
                        Some((Err(e), span)) => {
                            lex_error(&e, &span, &mut result_span, &mut errors);
                            None
                        }
                        Some((Ok(Token::LeftAngle), span)) => {
                            update_span(&mut result_span, &span);
                            Some(s!(x.into(), str_span))
                        }
                        x => {
                            expected_got_error_from_peek_result(
                                vec![TokenClass::RAngle],
                                x.as_ref(),
                                &mut result_span,
                                &mut errors,
                            );
                            None
                        }
                    },
                    x => {
                        expected_got_error_from_peek_result(
                            vec![TokenClass::Ident],
                            x.as_ref(),
                            &mut result_span,
                            &mut errors,
                        );
                        None
                    }
                }
            }
            _ => None,
        }
    };

    finish(
        s!(
            AttName { name, attrib },
            result_span.unwrap_or(SYNTHETIC_SPAN)
        ),
        errors,
    )
}

pub fn parse_ident_or_varargs<'a, E: Clone + fmt::Debug>(
    input: &mut Peekable<impl Iterator<Item = (Result<Token<'a>, E>, Span)>>,
) -> ParseResult<'a, Cow<'a, str>, E> {
    let mut result_span = None;
    let mut errors = vec![];

    let name = match input.next() {
        Some((Err(e), span)) => {
            lex_error(&e, &span, &mut result_span, &mut errors);
            "<error>".into()
        }
        Some((Ok(Token::Ident(x)), span)) => {
            update_span(&mut result_span, &span);
            x.into()
        }
        Some((Ok(Token::DotDotDot), span)) => {
            update_span(&mut result_span, &span);
            "<varargs>".into()
        }
        x => {
            expected_got_error_from_peek_result(
                vec![TokenClass::Ident],
                x.as_ref(),
                &mut result_span,
                &mut errors,
            );
            "<error>".into()
        }
    };

    finish(s!(name, result_span.unwrap_or(SYNTHETIC_SPAN)), errors)
}

pub fn parse_func_body<'a, E: Clone + fmt::Debug>(
    input: &mut Peekable<impl Iterator<Item = (Result<Token<'a>, E>, Span)>>,
) -> ParseResult<'a, FuncBody<'a>, E> {
    let mut result_span = None;
    let mut errors = vec![];

    let mut params = unbox(
        parse_list(
            input,
            Token::OParen,
            Token::CParen,
            Token::Comma,
            false,
            TokenClass::Ident,
            parse_ident_or_varargs,
        ),
        &mut result_span,
        &mut errors,
    );

    // Quick pass for varargs
    let mut varargs = None;
    for (i, x) in params.0.iter().enumerate() {
        if **x == "varargs" {
            varargs = Some(s!((), x.1.clone()));
            if i != params.len() - 1 {
                errors.push(ParseError::ExpectedGot {
                    span: x.1.clone(),
                    expected: vec![TokenClass::Ident],
                    got: Some(Token::DotDotDot),
                });
            }
        }
    }
    params.0.retain(|x| **x != "varargs");

    let block = unbox(parse_block(input), &mut result_span, &mut errors);

    match input.next() {
        Some((Err(e), span)) => {
            lex_error(&e, &span, &mut result_span, &mut errors);
            input.next();
        }
        Some((Ok(Token::End), _)) => {
            // good!
        }
        x => {
            // rip.
            expected_got_error_from_peek_result(
                vec![TokenClass::EndOfBlock],
                x.as_ref(),
                &mut result_span,
                &mut errors,
            );
        }
    }

    finish(
        s!(
            FuncBody {
                params,
                varargs,
                block
            },
            result_span.unwrap_or(SYNTHETIC_SPAN)
        ),
        errors,
    )
}

fn eat_then<'a, E: Clone + fmt::Debug>(
    input: &mut Peekable<impl Iterator<Item = (Result<Token<'a>, E>, Span)>>,
) -> ParseResult<'a, (), E> {
    let mut result_span = None;
    let mut errors = vec![];

    match input.peek() {
        Some((Err(e), span)) => {
            lex_error(e, span, &mut result_span, &mut errors);
            input.next();
        }
        Some((Ok(Token::Then), _)) => {
            // good!
            input.next();
        }
        x @ Some((Ok(Token::Do | Token::Colon), _)) => {
            // well. that's the wrong block start.
            expected_got_error_from_peek_result(
                vec![TokenClass::Then],
                x,
                &mut result_span,
                &mut errors,
            );
            input.next(); // but we're gonna assume they *intended* `then` and just eat the token anyway
        }
        x => {
            // annnnd they forgot entirely and just started the body. Oh well.
            expected_got_error_from_peek_result(
                vec![TokenClass::Then],
                x,
                &mut result_span,
                &mut errors,
            );
        }
    }

    finish(s!((), result_span.unwrap_or(SYNTHETIC_SPAN)), errors)
}

pub fn parse_stat<'a, E: Clone + fmt::Debug>(
    input: &mut Peekable<impl Iterator<Item = (Result<Token<'a>, E>, Span)>>,
) -> ParseResult<'a, Stat<'a>, E> {
    let mut result_span = None;
    let mut errors = vec![];

    let result_stat = s!(
        match input.peek() {
            Some((Err(e), span)) => {
                lex_error(e, span, &mut result_span, &mut errors);
                None
            }
            Some((Ok(Token::Semi), span)) => {
                update_span(&mut result_span, span);
                input.next();
                Some(Stat::Empty)
            }
            x @ Some((Ok(Token::Ident(_) | Token::OParen), span)) => {
                let first_result = x.cloned(); // used in case the expression is invalid
                update_span(&mut result_span, span);
                let first = unbox(parse_prefix_exp(input), &mut result_span, &mut errors);
                match first.0 {
                    PrefixExp::FunctionCall(x) => Some(Stat::FunctionCall(x)),
                    PrefixExp::Parens(_) => {
                        // TODO: make this error reference the whole expression rather than just the first token
                        expected_got_error_from_peek_result(
                            vec![TokenClass::Var],
                            first_result.as_ref(),
                            &mut result_span,
                            &mut errors,
                        );
                        None
                    }
                    PrefixExp::Var(x) => {
                        // Assign statement
                        // This'll be a bit of code duplication because we are handling a pair of "lists", but we're doing it way differently from what parse_list supports.
                        let mut vars = vec![x];

                        loop {
                            // We're expecting a comma first (or an assignment if the list of variables is done).
                            match input.next() {
                                Some((Err(e), span)) => {
                                    lex_error(&e, &span, &mut result_span, &mut errors);
                                }
                                Some((Ok(Token::Comma), span)) => {
                                    // yay!
                                    update_span(&mut result_span, &span);
                                }
                                Some((Ok(Token::Assign), _)) => {
                                    // also yay! we're done!
                                    // We're not updating the span on purpose. The variable span shouldn't include the assignment sigil, and the next expression will update the span when it appears.
                                    break;
                                }
                                x => {
                                    // uh oh.
                                    // So... this is *really* malformed code at this point. We're gonna just skip the rest of the list and jump to the equals sign.
                                    expected_got_error_from_peek_result(
                                        vec![TokenClass::Assign, TokenClass::Comma],
                                        x.as_ref(),
                                        &mut result_span,
                                        &mut errors,
                                    );
                                    recover(input, vec![Token::Assign]);
                                    input.next(); // annnnd consume the assign token.
                                    break; // annnnd get the heck out
                                }
                            }

                            // We got a comma; now, we need a variable.
                            let peeked = input.peek().cloned(); // used TEMPORARILY for error reporting if we got the wrong thing
                            let var = unbox(parse_prefix_exp(input), &mut result_span, &mut errors);

                            // Make sure it's actually a variable (since we funnel variable parsing through the prefixexp parser)
                            let var = match var.0 {
                                PrefixExp::Var(x) => Some(x), // yay!
                                _ => {
                                    // TODO: again, use the full expression in the error, not just the token.
                                    expected_got_error_from_peek_result(
                                        vec![TokenClass::Var],
                                        peeked.as_ref(),
                                        &mut result_span,
                                        &mut errors,
                                    );
                                    // Don't add this to the list; it's more trouble than it's worth.
                                    None
                                }
                            };
                            if let Some(var) = var {
                                vars.push(var);
                            }
                        }

                        let vars = s!(vars, result_span.clone().unwrap_or(SYNTHETIC_SPAN));

                        let mut exps_span = None; // Start this from scratch! We'll update the result span at the end.
                        let mut exps = vec![];

                        exps.push(unbox(parse_exp(input), &mut exps_span, &mut errors)); // ez

                        loop {
                            // Now, we're expecting a comma. If we don't find one, assume the statement is done.
                            match input.peek() {
                                Some((Err(e), span)) => {
                                    lex_error(e, span, &mut result_span, &mut errors);
                                    input.next();
                                }
                                Some((Ok(Token::Comma), span)) => {
                                    // yay!
                                    update_span(&mut result_span, span);
                                    input.next();
                                }
                                _ => {
                                    // annnnnd we're done. This is from the next statement, presumably, hence why we peeked earlier.
                                    break;
                                }
                            }

                            exps.push(unbox(parse_exp(input), &mut exps_span, &mut errors)); // again, ez
                        }

                        let exps = s!(exps, exps_span.unwrap_or(SYNTHETIC_SPAN));
                        update_span(&mut result_span, &exps.1);
                        Some(Stat::Assign { vars, exps })
                    }
                    PrefixExp::Error => {
                        None // Don't report an extra error, just keep recovering
                    }
                }
            }
            Some((Ok(Token::If), _)) => {
                input.next();
                let main_cond = unbox(parse_exp(input), &mut result_span, &mut errors);
                unbox(eat_then(input), &mut result_span, &mut errors);
                let main_block = unbox(parse_block(input), &mut result_span, &mut errors);
                let mut elseifs = Vec::new();
                let mut else_block = None;
                loop {
                    match input.next() {
                        Some((Err(e), span)) => {
                            lex_error(&e, &span, &mut result_span, &mut errors);
                        }
                        Some((Ok(Token::End), span)) => {
                            // yay! we're done!
                            update_span(&mut result_span, &span);
                            break;
                        }
                        x @ Some((Ok(Token::ElseIf), _)) => {
                            if else_block.is_some() {
                                // we already saw else; we should be done.
                                expected_got_error_from_peek_result(
                                    vec![TokenClass::EndOfBlock],
                                    x.as_ref(),
                                    &mut result_span,
                                    &mut errors,
                                );
                            }
                            // Alright, so we need an expression, a "then", and a block. Easy, right?
                            let cond = unbox(parse_exp(input), &mut result_span, &mut errors);
                            unbox(eat_then(input), &mut result_span, &mut errors);
                            let block = unbox(parse_block(input), &mut result_span, &mut errors);
                            elseifs.push((cond, Box::new(block)));
                        }
                        x @ Some((Ok(Token::Else), _)) => {
                            if else_block.is_some() {
                                // we already saw else; we should be done.
                                expected_got_error_from_peek_result(
                                    vec![TokenClass::EndOfBlock],
                                    x.as_ref(),
                                    &mut result_span,
                                    &mut errors,
                                );
                            }
                            else_block = Some(Box::new(unbox(
                                parse_block(input),
                                &mut result_span,
                                &mut errors,
                            )));
                        }
                        x => {
                            // ... how did we get here?
                            expected_got_error_from_peek_result(
                                vec![TokenClass::EndOfBlock],
                                x.as_ref(),
                                &mut result_span,
                                &mut errors,
                            );
                        }
                    }
                }
                Some(Stat::If {
                    main: (main_cond, Box::new(main_block)),
                    elseifs,
                    else_block,
                })
            }
            Some((Ok(Token::Local), _)) => {
                input.next();
                match input.peek() {
                    Some((Err(e), span)) => {
                        lex_error(e, span, &mut result_span, &mut errors);
                        None
                    }
                    Some((Ok(Token::Function), _)) => {
                        // Local function
                        input.next();

                        let name = unbox(parse_ident(input), &mut result_span, &mut errors);
                        let body = unbox(parse_func_body(input), &mut result_span, &mut errors);

                        Some(Stat::LocalFunction { name, body })
                    }
                    Some((Ok(Token::Ident(_)), _)) => {
                        // Local variable(s)
                        let mut names_span = None; // Start fresh
                        let mut names =
                            vec![unbox(parse_att_name(input), &mut names_span, &mut errors)];

                        let need_exps = loop {
                            match input.peek() {
                                Some((Err(e), span)) => {
                                    lex_error(e, span, &mut result_span, &mut errors);
                                    input.next();
                                    break false; // Bail from the statement, even if the user expected to assign things; the lexer error takes priority
                                }
                                Some((Ok(Token::Comma), _)) => {
                                    input.next();
                                }
                                Some((Ok(Token::Assign), _)) => {
                                    input.next();
                                    break true;
                                }
                                _ => {
                                    break false; // Statement over
                                }
                            }
                            names.push(unbox(parse_att_name(input), &mut names_span, &mut errors));
                        };

                        let names = s!(names, names_span.unwrap_or(SYNTHETIC_SPAN));

                        let exps = if need_exps {
                            let mut exps_span = None; // Start this from scratch! We'll update the result span at the end.
                            let mut exps = vec![];

                            exps.push(unbox(parse_exp(input), &mut exps_span, &mut errors)); // ez

                            loop {
                                // Now, we're expecting a comma. If we don't find one, assume the statement is done.
                                match input.peek() {
                                    Some((Err(e), span)) => {
                                        lex_error(e, span, &mut result_span, &mut errors);
                                        input.next();
                                    }
                                    Some((Ok(Token::Comma), span)) => {
                                        // yay!
                                        update_span(&mut result_span, span);
                                        input.next();
                                    }
                                    _ => {
                                        // annnnnd we're done. This is from the next statement, presumably, hence why we peeked earlier.
                                        break;
                                    }
                                }

                                exps.push(unbox(parse_exp(input), &mut exps_span, &mut errors)); // again, ez
                            }

                            let exps = s!(exps, exps_span.unwrap_or(SYNTHETIC_SPAN));
                            update_span(&mut result_span, &exps.1);
                            Some(exps)
                        } else {
                            None
                        };

                        Some(Stat::Local { names, exps })
                    }
                    x => {
                        expected_got_error_from_peek_result(
                            vec![TokenClass::Function, TokenClass::Ident],
                            x,
                            &mut result_span,
                            &mut errors,
                        );
                        None
                    }
                }
            }
            x => {
                expected_got_error_from_peek_result(
                    vec![TokenClass::Statement],
                    x,
                    &mut result_span,
                    &mut errors,
                );
                None
            }
        }
        .unwrap_or_else(|| {
            recover(
                input,
                vec![
                    Token::If,
                    Token::Else,
                    Token::ElseIf,
                    Token::Semi,
                    Token::End,
                ],
            );
            Stat::Error
        }),
        result_span.unwrap_or(SYNTHETIC_SPAN)
    );
    finish(result_stat, errors)
}

pub fn parse_block<'a, E: Clone + fmt::Debug>(
    input: &mut Peekable<impl Iterator<Item = (Result<Token<'a>, E>, Span)>>,
) -> ParseResult<'a, Block<'a>, E> {
    let mut result_span = None;

    let mut stats = vec![];
    let mut retstat = None;
    let mut errors = vec![];
    loop {
        match input.peek() {
            Some((Err(e), span)) => {
                lex_error(e, span, &mut result_span, &mut errors);
                input.next();
            }
            Some((Ok(Token::End | Token::Else | Token::ElseIf), _)) | None => break,
            Some((Ok(Token::Return), span)) => {
                let mut ret_span = Some(span.clone());
                input.next();
                let mut exps = vec![];
                match input.peek() {
                    Some((Err(e), span)) => {
                        lex_error(e, span, &mut result_span, &mut errors);
                        input.next();
                    }
                    Some((Ok(Token::End | Token::Else | Token::ElseIf), _)) | None => {}
                    Some((Ok(Token::Semi), _)) => {
                        input.next(); // skip semicolon
                    }
                    _ => {
                        exps.push(unbox(parse_exp(input), &mut result_span, &mut errors));
                    }
                }
                if !exps.is_empty() {
                    loop {
                        match input.peek() {
                            Some((Err(e), span)) => {
                                lex_error(e, span, &mut result_span, &mut errors);
                                input.next();
                                break;
                            }
                            Some((Ok(Token::Comma), _)) => {
                                input.next();
                            }
                            Some((Ok(Token::End | Token::Else | Token::ElseIf), _)) | None => break,
                            Some((Ok(Token::Semi), _)) => {
                                input.next(); // skip semicolon
                                break;
                            }
                            x => {
                                expected_got_error_from_peek_result(
                                    vec![
                                        TokenClass::Comma,
                                        TokenClass::EndOfBlock,
                                        TokenClass::Semi,
                                    ],
                                    x,
                                    &mut result_span,
                                    &mut errors,
                                );
                                break;
                            }
                        }
                        exps.push(unbox(parse_exp(input), &mut ret_span, &mut errors));
                    }
                }
                match input.peek() {
                    Some((Err(e), span)) => {
                        lex_error(e, span, &mut result_span, &mut errors);
                    }
                    Some((Ok(Token::End | Token::Else | Token::ElseIf), span)) => {
                        update_span(&mut result_span, span);
                    }
                    None => {}
                    x => {
                        expected_got_error_from_peek_result(
                            vec![TokenClass::EndOfBlock],
                            x,
                            &mut ret_span,
                            &mut errors,
                        );
                    }
                }
                retstat = Some(s!(exps, ret_span.unwrap()));
                break;
            }
            _ => {
                stats.push(unbox(parse_stat(input), &mut result_span, &mut errors));
            }
        }
    }

    let result_block = s!(
        Block { stats, retstat },
        result_span.unwrap_or(SYNTHETIC_SPAN)
    );
    finish(result_block, errors)
}
