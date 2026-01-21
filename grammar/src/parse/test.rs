use crate::{
    ast::{Args, Block, Exp, FunctionCall, PrefixExp, Var},
    lex::Token,
    parse::parse_block,
    s,
};

#[test]
fn hello_world() {
    let tokens = [
        (Token::Ident("print"), 0..5),
        (Token::OParen, 5..6),
        (Token::StringLiteral(r#""Hello World""#), 6..19),
        (Token::CParen, 19..20),
    ];
    let result = parse_block(
        &mut tokens
            .into_iter()
            .map(|(a, b)| (Ok::<_, ()>(a), b))
            .peekable(),
    );
    assert_eq!(
        result,
        Ok(s!(
            Block {
                stats: vec![s!(
                    crate::ast::Stat::FunctionCall(s!(
                        FunctionCall {
                            lhs: Box::new(s!(
                                PrefixExp::Var(s!(Var::Name(s!("print".into(), 0..5)), 0..5)),
                                0..5
                            )),
                            method: None,
                            args: s!(
                                Args::List(s!(
                                    vec![s!(
                                        Exp::LiteralString(s!(Box::new(*b"Hello World"), 6..19)),
                                        6..19
                                    )],
                                    5..20
                                )),
                                5..20
                            ),
                        },
                        0..20
                    )),
                    0..20
                )],
                retstat: None
            },
            0..20
        ))
    );
}
