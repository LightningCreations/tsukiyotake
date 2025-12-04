use logos::{Logos as _, SpannedIter};

use crate::lex::Token;

pub type LalrpopSpanned<Tok> = Result<(usize, Tok, usize), ()>;

pub struct Lexer<'src> {
    token_stream: SpannedIter<'src, Token<'src>>,
}

impl<'src> Lexer<'src> {
    pub fn new(input: &'src str) -> Self {
        Self {
            token_stream: Token::lexer(input).spanned(),
        }
    }
}

impl<'src> Iterator for Lexer<'src> {
    type Item = LalrpopSpanned<Token<'src>>;

    fn next(&mut self) -> Option<Self::Item> {
        self.token_stream
            .next()
            .map(|(token, span)| Ok((span.start, token?, span.end)))
    }
}
