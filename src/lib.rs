#![cfg_attr(not(feature = "std"), no_std)]
#![feature(
    bstr,
    macro_derive,
    allocator_api,
    push_mut,
    maybe_uninit_fill,
    try_trait_v2,
    int_roundings,
    yeet_expr
)]

extern crate alloc;

#[macro_use]
mod macros;

pub mod hir;

pub use tsukiyotake_grammar::{Logos, Span, ast, lex, parse};

pub mod engine;
pub mod sync;

pub mod mir;
