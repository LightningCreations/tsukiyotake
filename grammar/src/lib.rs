#![allow(incomplete_features)] // Remove these if they're breaking more than they're fixing
#![feature(lazy_type_alias)]

extern crate alloc;

pub mod ast;
pub mod lex;
pub mod parse;

pub use logos::Logos;
pub use logos::Span;
