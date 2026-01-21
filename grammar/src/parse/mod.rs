#[cfg(test)]
mod test;

use alloc::fmt;
use core::iter::Peekable;

use logos::Span;

use crate::{
    ast::{
        Args, BinOp, Block, Exp, FunctionCall, PrefixExp, SYNTHETIC_POS, SYNTHETIC_SPAN, Spanned,
        Stat, UnOp, Var,
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

    False,
    True,

    Ident,
    Number,
    StringLiteral,

    OParen,
    CParen,
    OBrace,
    CBrace,
    OSquare,
    CSquare,

    Expression,
    EndOfBlock,
    Eof,
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
            None => break,
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
            }
        }
    }

    finish(s!(list, result_span.unwrap_or(SYNTHETIC_SPAN)), errors)
}

fn parse_atom<'a, E: Clone + fmt::Debug>(
    input: &mut Peekable<impl Iterator<Item = (Result<Token<'a>, E>, Span)>>,
) -> ParseResult<'a, Exp<'a>, E> {
    let mut result_span = None;
    let mut errors = vec![];

    let result_atom = match input.peek() {
        Some((Ok(Token::StringLiteral(x)), span)) => {
            update_span(&mut result_span, span);
            let str = parse_string(x);
            let span = span.clone();
            input.next();
            Some(Exp::LiteralString(s!(str, span)))
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

    // TODO: Actual Pratt parsing

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
        Some((Ok(Token::OParen), _)) => todo!(),
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
                ))
            }
            Some((Ok(Token::Dot), _)) => todo!(),
            Some((Ok(Token::OSquare), _)) => todo!(),
            _ => {
                break; // not part of a prefixexp anymore; get out
            }
        }
    }
    finish(s!(lhs, result_span.unwrap_or(SYNTHETIC_SPAN)), errors)
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
                        // assign
                        todo!()
                    }
                    PrefixExp::Error => {
                        None // Don't report an extra error, just keep recovering
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
            Some((Ok(Token::Return), _)) => todo!("return statement"),
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
