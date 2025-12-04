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
mod grammar_test;
