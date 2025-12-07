extern crate alloc;

use lalrpop_util::lalrpop_mod;

pub mod ast;
pub mod lex;
pub mod logos_lalrpop_bridge;
lalrpop_mod!(pub grammar);

#[cfg(test)]
mod grammar_test;

pub use logos::Span;
