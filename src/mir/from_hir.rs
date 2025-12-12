pub struct MirConverter {
    
}

#[cfg(test)]
mod test {
    use alloc::{boxed::Box, vec};
    use hashbrown::HashSet;
    use tsukiyotake_grammar::{s, synth};

    use crate::hir;

    fn hello_world() {
        let hir = hir::Block {
            locals: HashSet::new(),
            import_locals: HashSet::new(),
            stats: vec![s!(
                hir::Stat::FunctionCall(s!(
                    hir::FunctionCall {
                        lhs: Box::new(s!(
                            hir::Exp::Var(s!(
                                hir::Var::Path {
                                    lhs: Box::new(synth!(hir::Exp::Var(synth!(hir::Var::Name(
                                        synth!("_ENV".into())
                                    ))))),
                                    field: s!("print", 0..5)
                                },
                                0..5,
                            )),
                            0..5,
                        )),
                        args: s!(
                            vec![s!(
                                hir::Exp::LiteralString(s!(Box::new(*br#"Hello World"#), 6..19)),
                                6..19,
                            )],
                            6..19,
                        )
                    },
                    0..20,
                )),
                0..20,
            )],
            retstat: None,
        };
    }
}
