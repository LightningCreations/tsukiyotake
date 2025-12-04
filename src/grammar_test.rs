use crate::{ast::*, grammar::BlockParser, logos_lalrpop_bridge::Lexer};
use alloc::{boxed::Box, vec};
use indoc::indoc;

#[test]
fn hello_world() {
    let input = r#"print("Hello World")"#;
    let lexer = Lexer::new(&input);
    let parser = BlockParser::new();
    let ast = parser.parse(input, lexer);
    assert_eq!(
        ast,
        Ok(Block {
            stats: vec![(
                Stat::FunctionCall((
                    FunctionCall {
                        lhs: Box::new((PrefixExp::Var((Var::Name(("print", 0..5)), 0..5)), 0..5)),
                        method: None,
                        args: (
                            Args::List((
                                vec![(
                                    Exp::LiteralString((Box::new(*br#""Hello World""#), 6..19)),
                                    6..19
                                )],
                                6..19
                            )),
                            5..20
                        )
                    },
                    0..20
                )),
                0..20
            )],
            retstat: None,
        })
    );
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
    let lexer = Lexer::new(&input);
    let parser = BlockParser::new();
    let ast = parser.parse(input, lexer);
    #[rustfmt::skip] // I put too much time into this formatting
    assert_eq!(
        ast,
        Ok(Block {
            stats: vec![
                (Stat::Function {
                    name: (FuncName {
                        path: (vec![
                            ("fact", 41..45)
                        ], 41..45),
                        method: None
                    }, 41..45),
                    body: (FuncBody {
                        params: (vec![
                            ("n", 46..47)
                        ], 46..47),
                        varargs: None,
                        block: (Block {
                            stats: vec![
                                (Stat::If {
                                    main: (
                                        (Exp::BinExp {
                                            lhs: Box::new(
                                                (Exp::PrefixExp((
                                                    PrefixExp::Var(
                                                        (Var::Name(
                                                            ("n", 56..57)
                                                        ), 56..57)
                                                    ), 56..57)
                                                ), 56..57)
                                            ),
                                            op: BinOp::Eq,
                                            rhs: Box::new(
                                                (Exp::NumeralInt(
                                                    (0, 61..62)
                                                ), 61..62)
                                            ),
                                        }, 56..62),
                                        Box::new(
                                            (Block {
                                                stats: vec![],
                                                retstat: Some((vec![
                                                    (Exp::NumeralInt((1, 83..84)), 83..84)
                                                ], 83..84))
                                            }, 76..84)
                                        ),
                                    ),
                                    elseifs: vec![],
                                    else_block: Some(Box::new(
                                        (Block {
                                            stats: vec![],
                                            retstat: Some((vec![
                                                (Exp::BinExp {
                                                    lhs: Box::new(
                                                        (Exp::PrefixExp(
                                                            (PrefixExp::Var(
                                                                (Var::Name(
                                                                    ("n", 109..110)
                                                                ), 109..110)
                                                            ), 109..110)
                                                        ), 109..110)
                                                    ),
                                                    op: BinOp::Mul,
                                                    rhs: Box::new(
                                                        (Exp::PrefixExp((
                                                            PrefixExp::FunctionCall(
                                                                (FunctionCall {
                                                                    lhs: Box::new(
                                                                        (PrefixExp::Var(
                                                                            (Var::Name(
                                                                                ("fact", 113..117)
                                                                            ), 113..117)
                                                                        ), 113..117)
                                                                    ),
                                                                    method: None,
                                                                    args: (Args::List(
                                                                        (vec![
                                                                            (Exp::BinExp {
                                                                                lhs: Box::new(
                                                                                    (Exp::PrefixExp(
                                                                                        (PrefixExp::Var(
                                                                                            (Var::Name(
                                                                                                ("n", 118..119)
                                                                                            ), 118..119)
                                                                                        ), 118..119)
                                                                                    ), 118..119)
                                                                                ),
                                                                                op: BinOp::Sub,
                                                                                rhs: Box::new(
                                                                                    (Exp::NumeralInt(
                                                                                        (1, 120..121)
                                                                                    ), 120..121)
                                                                                ),
                                                                            }, 118..121)
                                                                        ], 118..121)
                                                                    ), 117..122)
                                                                }, 113..122)
                                                            ), 113..122)
                                                        ), 113..122)
                                                    )
                                                }, 109..122)
                                            ], 109..122))
                                        }, 102..122)
                                    ))
                                }, 53..130)
                            ],
                            retstat: None
                        }, 53..130)
                    }, 45..134),
                }, 32..134),
                (Stat::FunctionCall(
                    (FunctionCall {
                        lhs: Box::new(
                            (PrefixExp::Var(
                                (Var::Name(
                                    ("print", 136..141)
                                ), 136..141)
                            ), 136..141)
                        ),
                        method: None,
                        args: (Args::List(
                            (vec![
                                (Exp::LiteralString(
                                    (Box::new(*br#""enter a number:""#), 142..159)
                                ), 142..159)
                            ], 142..159)
                        ), 141..160)
                    }, 136..160)
                ), 136..160),
                (Stat::Assign {
                    vars: (vec![
                        (Var::Name(
                            ("a", 161..162)
                        ), 161..162)
                    ], 161..162),
                    exps: (vec![
                        (Exp::PrefixExp(
                            (PrefixExp::FunctionCall(
                                (FunctionCall {
                                    lhs: Box::new(
                                        (PrefixExp::Var(
                                            (Var::Path {
                                                lhs: Box::new(
                                                    (PrefixExp::Var(
                                                        (Var::Name(
                                                            ("io", 165..167)
                                                        ), 165..167)
                                                    ), 165..167)
                                                ),
                                                member: ("read", 168..172)
                                            }, 165..172)
                                        ), 165..172)
                                    ),
                                    method: None,
                                    args: (Args::List(
                                        (vec![
                                            (Exp::LiteralString(
                                                (Box::new(*br#""*number""#), 173..182)
                                            ), 173..182)
                                        ], 173..182)
                                    ), 172..183)
                                }, 165..183)
                            ), 165..183)
                        ), 165..183)
                    ], 165..183)
                }, 161..183),
                (Stat::FunctionCall(
                    (FunctionCall {
                        lhs: Box::new(
                            (PrefixExp::Var(
                                (Var::Name(
                                    ("print", 184..189)
                                ), 184..189)
                            ), 184..189)
                        ),
                        method: None,
                        args: (Args::List(
                            (vec![
                                (Exp::PrefixExp(
                                    (PrefixExp::FunctionCall(
                                        (FunctionCall {
                                            lhs: Box::new((PrefixExp::Var(
                                                (Var::Name(
                                                    ("fact", 190..194)
                                                ), 190..194)
                                            ), 190..194)),
                                            method: None,
                                            args: (Args::List(
                                                (vec![
                                                    (Exp::PrefixExp(
                                                        (PrefixExp::Var(
                                                            (Var::Name(
                                                                ("a", 195..196)
                                                            ), 195..196)
                                                        ), 195..196)
                                                    ), 195..196)
                                                ], 195..196)
                                            ), 194..197)
                                        }, 190..197)
                                    ), 190..197)
                                ), 190..197)
                            ], 190..197)
                        ), 189..198)
                    }, 184..198)
                ), 184..198)
            ], retstat: None
        })
    );
}
