use crate::{ast::*, grammar::BlockParser, logos_lalrpop_bridge::Lexer, s};
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
            stats: vec![s!(
                Stat::FunctionCall(s!(
                    FunctionCall {
                        lhs: Box::new(s!(
                            PrefixExp::Var(s!(Var::Name(s!("print", 0..5)), 0..5)),
                            0..5
                        )),
                        method: None,
                        args: s!(
                            Args::List(s!(
                                vec![s!(
                                    Exp::LiteralString(s!(Box::new(*b"Hello World"), 6..19)),
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
                s!(Stat::Function {
                    name: s!(FuncName {
                        path: s!(vec![
                            s!("fact", 41..45)
                        ], 41..45),
                        method: None
                    }, 41..45),
                    body: s!(FuncBody {
                        params: s!(vec![
                            s!("n", 46..47)
                        ], 46..47),
                        varargs: None,
                        block: s!(Block {
                            stats: vec![
                                s!(Stat::If {
                                    main: (
                                        s!(Exp::BinExp {
                                            lhs: Box::new(
                                                s!(Exp::PrefixExp(s!(
                                                    PrefixExp::Var(
                                                        s!(Var::Name(
                                                            s!("n", 56..57)
                                                        ), 56..57)
                                                    ), 56..57)
                                                ), 56..57)
                                            ),
                                            op: BinOp::Eq,
                                            rhs: Box::new(
                                                s!(Exp::NumeralInt(
                                                    s!(0, 61..62)
                                                ), 61..62)
                                            ),
                                        }, 56..62),
                                        Box::new(
                                            s!(Block {
                                                stats: vec![],
                                                retstat: Some(s!(vec![
                                                    s!(Exp::NumeralInt(s!(1, 83..84)), 83..84)
                                                ], 83..84))
                                            }, 76..84)
                                        ),
                                    ),
                                    elseifs: vec![],
                                    else_block: Some(Box::new(
                                        s!(Block {
                                            stats: vec![],
                                            retstat: Some(s!(vec![
                                                s!(Exp::BinExp {
                                                    lhs: Box::new(
                                                        s!(Exp::PrefixExp(
                                                            s!(PrefixExp::Var(
                                                                s!(Var::Name(
                                                                    s!("n", 109..110)
                                                                ), 109..110)
                                                            ), 109..110)
                                                        ), 109..110)
                                                    ),
                                                    op: BinOp::Mul,
                                                    rhs: Box::new(
                                                        s!(Exp::PrefixExp(s!(
                                                            PrefixExp::FunctionCall(
                                                                s!(FunctionCall {
                                                                    lhs: Box::new(
                                                                        s!(PrefixExp::Var(
                                                                            s!(Var::Name(
                                                                                s!("fact", 113..117)
                                                                            ), 113..117)
                                                                        ), 113..117)
                                                                    ),
                                                                    method: None,
                                                                    args: s!(Args::List(
                                                                        s!(vec![
                                                                            s!(Exp::BinExp {
                                                                                lhs: Box::new(
                                                                                    s!(Exp::PrefixExp(
                                                                                        s!(PrefixExp::Var(
                                                                                            s!(Var::Name(
                                                                                                s!("n", 118..119)
                                                                                            ), 118..119)
                                                                                        ), 118..119)
                                                                                    ), 118..119)
                                                                                ),
                                                                                op: BinOp::Sub,
                                                                                rhs: Box::new(
                                                                                    s!(Exp::NumeralInt(
                                                                                        s!(1, 120..121)
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
                s!(Stat::FunctionCall(
                    s!(FunctionCall {
                        lhs: Box::new(
                            s!(PrefixExp::Var(
                                s!(Var::Name(
                                    s!("print", 136..141)
                                ), 136..141)
                            ), 136..141)
                        ),
                        method: None,
                        args: s!(Args::List(
                            s!(vec![
                                s!(Exp::LiteralString(
                                    s!(Box::new(*b"enter a number:"), 142..159)
                                ), 142..159)
                            ], 142..159)
                        ), 141..160)
                    }, 136..160)
                ), 136..160),
                s!(Stat::Assign {
                    vars: s!(vec![
                        s!(Var::Name(
                            s!("a", 161..162)
                        ), 161..162)
                    ], 161..162),
                    exps: s!(vec![
                        s!(Exp::PrefixExp(
                            s!(PrefixExp::FunctionCall(
                                s!(FunctionCall {
                                    lhs: Box::new(
                                        s!(PrefixExp::Var(
                                            s!(Var::Path {
                                                lhs: Box::new(
                                                    s!(PrefixExp::Var(
                                                        s!(Var::Name(
                                                            s!("io", 165..167)
                                                        ), 165..167)
                                                    ), 165..167)
                                                ),
                                                member: s!("read", 168..172)
                                            }, 165..172)
                                        ), 165..172)
                                    ),
                                    method: None,
                                    args: s!(Args::List(
                                        s!(vec![
                                            s!(Exp::LiteralString(
                                                s!(Box::new(*b"*number"), 173..182)
                                            ), 173..182)
                                        ], 173..182)
                                    ), 172..183)
                                }, 165..183)
                            ), 165..183)
                        ), 165..183)
                    ], 165..183)
                }, 161..183),
                s!(Stat::FunctionCall(
                    s!(FunctionCall {
                        lhs: Box::new(
                            s!(PrefixExp::Var(
                                s!(Var::Name(
                                    s!("print", 184..189)
                                ), 184..189)
                            ), 184..189)
                        ),
                        method: None,
                        args: s!(Args::List(
                            s!(vec![
                                s!(Exp::PrefixExp(
                                    s!(PrefixExp::FunctionCall(
                                        s!(FunctionCall {
                                            lhs: Box::new(s!(PrefixExp::Var(
                                                s!(Var::Name(
                                                    s!("fact", 190..194)
                                                ), 190..194)
                                            ), 190..194)),
                                            method: None,
                                            args: s!(Args::List(
                                                s!(vec![
                                                    s!(Exp::PrefixExp(
                                                        s!(PrefixExp::Var(
                                                            s!(Var::Name(
                                                                s!("a", 195..196)
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

#[test]
fn factorial_fmt() {
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
    let ast = parser.parse(input, lexer).unwrap();
    assert_eq!(ast.to_string(), indoc! {r#"
        function fact(n)
            if n == 0 then
                return 1
            else
                return n * fact(n - 1)
            end
        end
        print("enter a number:")
        a = io.read("*number")
        print(fact(a))
    "#});
}
