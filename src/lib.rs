#![cfg_attr(not(feature = "std"), no_std)]
#![feature(macro_derive)]

extern crate alloc;

#[macro_use]
mod macros;

pub mod hir;

pub use tsukiyotake_grammar::{ast, grammar, lex, logos_lalrpop_bridge};

pub mod engine;
pub mod sync;

pub mod mir;
