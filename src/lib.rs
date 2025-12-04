#![cfg_attr(not(feature = "std"), no_std)]
#![feature(macro_derive)]

use lalrpop_util::lalrpop_mod;

extern crate alloc;

pub mod ast;
pub mod lex;
pub mod logos_lalrpop_bridge;

lalrpop_mod!(grammar);

pub mod engine;
pub mod sync;

#[cfg(test)]
mod grammar_test {
    use crate::{ast::*, grammar::BlockParser, logos_lalrpop_bridge::Lexer};
    use alloc::{boxed::Box, vec};

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
                            lhs: Box::new((
                                PrefixExp::Var((Var::Name(("print", 0..5)), 0..5)),
                                0..5
                            )),
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
}
