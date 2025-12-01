use logos::{Lexer, Logos, Span};

#[derive(Logos, Debug, PartialEq)]
#[logos(error(Span, |lex| lex.span()))]
pub enum Token {
    #[token("and")]
    And,
    #[token("break")]
    Break,
    #[token("do")]
    Do,
    #[token("else")]
    Else,
    #[token("elseif")]
    ElseIf,
    #[token("end")]
    End,
    #[token("false")]
    False,
    #[token("for")]
    For,
    #[token("function")]
    Function,
    #[token("goto")]
    Goto,
    #[token("if")]
    If,
    #[token("in")]
    In,
    #[token("local")]
    Local,
    #[token("nil")]
    Nil,
    #[token("not")]
    Not,
    #[token("or")]
    Or,
    #[token("repeat")]
    Repeat,
    #[token("return")]
    Return,
    #[token("then")]
    Then,
    #[token("true")]
    True,
    #[token("until")]
    Until,
    #[token("while")]
    While,
    #[regex("[A-Za-z_][A-Za-z0-9_]*", |lexer| lexer.span())]
    Ident(Span),
    #[token("+")]
    Plus,
    #[token("-")]
    Minus,
    #[token("*")]
    Star,
    #[token("/")]
    Slash,
    #[token("%")]
    Modulo,
    #[token("^")]
    Hat,
    #[token("#")]
    Count,
    #[token("&")]
    BitAnd,
    #[token("~")]
    BitNot,
    #[token("|")]
    BitOr,
    #[token("<<")]
    LeftShift,
    #[token(">>")]
    RightShift,
    #[token("//")]
    IntDiv,
    #[token("==")]
    Equals,
    #[token("~=")]
    NotEquals,
    #[token("<=")]
    LessEquals,
    #[token(">=")]
    GreaterEquals,
    #[token("<")]
    LeftAngle,
    #[token(">")]
    RightAngle,
    #[token("=")]
    Assign,
    #[token("(")]
    OParen,
    #[token(")")]
    CParen,
    #[token("{")]
    OBrace,
    #[token("}")]
    CBrace,
    #[token("[")]
    OSquare,
    #[token("]")]
    CSquare,
    #[token("::")]
    ColonColon,
    #[token(";")]
    Semi,
    #[token(":")]
    Colon,
    #[token(",")]
    Comma,
    #[token(".")]
    Dot,
    #[token("..")]
    DotDot,
    #[token("...")]
    DotDotDot,

    #[regex("[1-9][0-9]*(\\.[0-9]+)?([eE][1-9][0-9]*)?", |lexer| lexer.span())]
    #[regex("0x[0-9A-Fa-f]*(\\.[0-9A-Fa-f]*)?([pP][1-9][0-9]*)?", |lexer| lexer.span())]
    Number(Span),

    #[regex("\"([^\"\\\n\r]|(\\\\[\\\\nr'\"abtv])|(\\\\u\\{[0-9A-Fa-f]{1,8}\\})|(\\\\z[[:space:]]*)|(\\\\[0-9]{1,3})|(\\\\x[0-9A-Fa-f]{2}))\"", |lexer| lexer.span())]
    StringLiteral(Span),

    #[regex("\\[=*\\[", parse_raw_string)]
    RawString(Span),

    #[regex("[[:space:]]+", logos::skip)]
    #[regex("--[\\[\\n]*\\n", logos::skip)]
    #[regex("--\\[[^\\]]*\\]", logos::skip)]
    Whitespace,
}

fn parse_raw_string(lex: &mut Lexer<Token>) -> Result<Span, Span> {
    let n = lex.slice().len().strict_sub(2);

    let mut rem = lex.remainder();
    let mut total_len = 0;

    'a: while let Some(x) = rem.find('[') {
        total_len += x;

        let l = &rem[x..];

        rem = l;

        let b = l.as_bytes();

        if b.len() < (n + 2) {
            lex.bump(total_len);
            return Err(lex.span());
        }

        for b in b.iter().skip(1).take(n) {
            if *b != b'=' {
                continue 'a;
            }
        }

        if b[1 + n] == b']' {
            lex.bump(total_len + 2 + n);
            return Ok(lex.span());
        }
    }

    lex.bump(total_len + rem.len());
    Err(lex.span())
}

#[cfg(test)]
mod test {
    use alloc::vec;
    use alloc::vec::Vec;
    use logos::{Lexer, Span};

    use crate::lex::Token;

    #[test]
    pub fn hello_world() {
        let input = r#"print("Hello World")"#;
        let lexed: Vec<(Token, Span)> = Lexer::new(input)
            .spanned()
            .map(|(x, y)| (x.unwrap(), y))
            .collect();

        assert_eq!(
            lexed,
            vec![
                (Token::Ident(0..5), 0..5),
                (Token::OParen, 5..6),
                (Token::StringLiteral(6..19), 6..19),
                (Token::CParen, 19..20)
            ]
        );
    }
}
