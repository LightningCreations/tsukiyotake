use alloc::{boxed::Box, string::String, vec::Vec};
use logos::{Lexer, Span};

use crate::{ast::Block, lex::Token};

pub struct ParseError {
    error: String,
    span: Span,
}

pub type Result<T> = core::result::Result<T, ParseError>;

pub fn parse_chunk<'src>(lexer: &'src mut Lexer<'src, Token<'src>>) -> Result<Block<'src>> {
    todo!()
}
