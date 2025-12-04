use core::fmt;

use logos::{Lexer, Logos};

#[derive(Logos, Debug, Clone, PartialEq)]
// #[logos(export_dir="diagrams")]
pub enum Token<'src> {
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
    #[regex("[A-Za-z_][A-Za-z0-9_]*", |lexer| lexer.slice())]
    Ident(&'src str),
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

    #[regex("[1-9][0-9]*(\\.[0-9]+)?([eE][1-9][0-9]*)?", |lexer| lexer.slice())]
    #[regex("0x[0-9A-Fa-f]*(\\.[0-9A-Fa-f]*)?([pP][1-9][0-9]*)?", |lexer| lexer.slice())]
    Number(&'src str),

    #[regex("\"([^\"\\\n\r]|(\\\\[\\\\nr'\"abtv])|(\\\\u\\{[0-9A-Fa-f]{1,8}\\})|(\\\\z[[:space:]]*)|(\\\\[0-9]{1,3})|(\\\\x[0-9A-Fa-f]{2}))*\"", |lexer| lexer.slice())]
    StringLiteral(&'src str),

    #[regex("\\[=*\\[", parse_raw_string)]
    RawString(&'src str),

    #[regex("[[:space:]]+", logos::skip)]
    #[regex("--[\\[\\n]*\\n", logos::skip)]
    #[regex("--\\[[^\\]]*\\]", logos::skip)]
    Whitespace,
}

impl fmt::Display for Token<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Token::And => write!(f, "and"),
            Token::Break => write!(f, "break"),
            Token::Do => write!(f, "do"),
            Token::Else => write!(f, "else"),
            Token::ElseIf => write!(f, "elseif"),
            Token::End => write!(f, "end"),
            Token::False => write!(f, "false"),
            Token::For => write!(f, "for"),
            Token::Function => write!(f, "function"),
            Token::Goto => write!(f, "goto"),
            Token::If => write!(f, "if"),
            Token::In => write!(f, "in"),
            Token::Local => write!(f, "local"),
            Token::Nil => write!(f, "nil"),
            Token::Not => write!(f, "not"),
            Token::Or => write!(f, "or"),
            Token::Repeat => write!(f, "repeat"),
            Token::Return => write!(f, "return"),
            Token::Then => write!(f, "then"),
            Token::True => write!(f, "true"),
            Token::Until => write!(f, "until"),
            Token::While => write!(f, "while"),
            Token::Ident(s) => write!(f, "{s}"),
            Token::Plus => write!(f, "+"),
            Token::Minus => write!(f, "-"),
            Token::Star => write!(f, "*"),
            Token::Slash => write!(f, "/"),
            Token::Modulo => write!(f, "%"),
            Token::Hat => write!(f, "^"),
            Token::Count => write!(f, "#"),
            Token::BitAnd => write!(f, "&"),
            Token::BitNot => write!(f, "~"),
            Token::BitOr => write!(f, "|"),
            Token::LeftShift => write!(f, "<<"),
            Token::RightShift => write!(f, ">>"),
            Token::IntDiv => write!(f, "//"),
            Token::Equals => write!(f, "=="),
            Token::NotEquals => write!(f, "~="),
            Token::LessEquals => write!(f, "<="),
            Token::GreaterEquals => write!(f, ">="),
            Token::LeftAngle => write!(f, "<"),
            Token::RightAngle => write!(f, ">"),
            Token::Assign => write!(f, "="),
            Token::OParen => write!(f, "("),
            Token::CParen => write!(f, ")"),
            Token::OBrace => write!(f, "{{"),
            Token::CBrace => write!(f, "}}"),
            Token::OSquare => write!(f, "["),
            Token::CSquare => write!(f, "]"),
            Token::ColonColon => write!(f, "::"),
            Token::Semi => write!(f, ";"),
            Token::Colon => write!(f, ":"),
            Token::Comma => write!(f, ","),
            Token::Dot => write!(f, "."),
            Token::DotDot => write!(f, ".."),
            Token::DotDotDot => write!(f, "..."),
            Token::Number(s) => write!(f, "{s}"),
            Token::StringLiteral(s) => write!(f, "{s}"),
            Token::RawString(s) => write!(f, "{s}"),
            Token::Whitespace => write!(f, "<whitespace>"),
        }
    }
}

fn parse_raw_string<'src>(lex: &mut Lexer<'src, Token<'src>>) -> Result<&'src str, ()> {
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
            return Err(());
        }

        for b in b.iter().skip(1).take(n) {
            if *b != b'=' {
                continue 'a;
            }
        }

        if b[1 + n] == b']' {
            lex.bump(total_len + 2 + n);
            return Ok(lex.slice());
        }
    }

    lex.bump(total_len + rem.len());
    Err(())
}

#[cfg(test)]
mod test {
    use indoc::indoc;
    use logos::Logos;

    use crate::lex::Token;

    #[test]
    fn hello_world() {
        let input = r#"print("Hello World")"#;
        let mut lexer = Token::lexer(input).spanned();

        assert_eq!(lexer.next(), Some((Ok(Token::Ident("print")), 0..5)));
        assert_eq!(lexer.next(), Some((Ok(Token::OParen), 5..6)));
        assert_eq!(
            lexer.next(),
            Some((Ok(Token::StringLiteral(r#""Hello World""#)), 6..19)),
        );
        assert_eq!(lexer.next(), Some((Ok(Token::CParen), 19..20)));
        assert_eq!(lexer.next(), None);
    }

    #[test]
    fn single_comment() {
        let input = "-- this is a comment\n";
        let mut lexer = Token::lexer(input).spanned();

        assert_eq!(lexer.next(), None);
    }

    #[test]
    fn factorial() {
        let input = indoc! {r#"
            -- defines a factorial function
            function fact(n)
                if n == 0 then
                    return 1
                else
                    return n * fact(n-1)
                end
            end

            print("enter a number:")
            a = io.read("*number")
            print(fact(a))
        "#};
        let mut lexer = Token::lexer(input).spanned();

        assert_eq!(lexer.next(), Some((Ok(Token::Function), 32..40)));
    }
}
