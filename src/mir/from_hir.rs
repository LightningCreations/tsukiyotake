use hashbrown::HashMap;

use crate::{bb, hir, mir::*};

#[derive(Debug)]
struct UnbuiltBasicBlock {
    id: BasicBlockId,
    stats: Vec<Statement>,
    term: Option<Terminator>,
}

impl UnbuiltBasicBlock {
    fn new() -> Self {
        Self {
            id: BasicBlockId::new_unchecked(NonZeroU32::new(1).unwrap()),
            stats: Vec::new(),
            term: None,
        }
    }

    fn finish_and_reset(&mut self) -> BasicBlock {
        BasicBlock {
            id: self.id,
            stats: core::mem::take(&mut self.stats),
            term: core::mem::take(&mut self.term),
        }
    }

    fn push(&mut self, stat: Statement) {
        self.stats.push(stat);
    }
}

#[derive(Debug)]
pub struct MirConverter {
    debug_info: FunctionDebugInfo,
    vars_by_name: HashMap<String, SsaVarId>,
    num_upvars: u32,
    num_params: u32,
    variadic: bool,
    basic_blocks: Vec<BasicBlock>,
    cur_block: UnbuiltBasicBlock,
    next_var: SsaVarId,
    next_block: BasicBlockId,
}

impl MirConverter {
    pub fn new(
        name: Spanned<String>,
        upvars: &[Spanned<String>],
        params: &[Spanned<String>],
        variadic: bool,
    ) -> Self {
        let mut result = Self {
            debug_info: FunctionDebugInfo {
                var_names_map: BTreeMap::new(),
                function_canon_name: name,
            },
            vars_by_name: HashMap::new(),
            num_upvars: upvars.len() as u32,
            num_params: params.len() as u32,
            variadic,
            basic_blocks: Vec::new(),
            cur_block: UnbuiltBasicBlock::new(),
            next_var: SsaVarId::_ENV,
            next_block: bb!(1),
        };

        if !upvars.is_empty() {
            todo!("upvars are interesting")
        }

        result.add_var(Spanned::synth("_ENV".into()));
        for param in params {
            result.add_var(param.clone());
        }
        if variadic {
            result.add_var(Spanned::synth("...".into()));
        }

        result
    }

    fn add_var(&mut self, name: Spanned<String>) -> SsaVarId {
        let var = self.next_var;
        self.vars_by_name.insert(name.0.clone(), var);
        self.debug_info.var_names_map.insert(var, name);
        self.next_var = var.next();
        var
    }

    pub fn new_at_root() -> Self {
        // Handles the root chunk, as it has very particular settings
        Self::new(Spanned::synth("<root>".into()), &[], &[], true)
    }

    fn get_var(&mut self, name: &str) -> SsaVarId {
        if let Some(x) = self.vars_by_name.get(name) {
            *x
        } else {
            todo!()
        }
    }

    fn convert_exp(&mut self, exp: &hir::Exp) -> Expr {
        match exp {
            hir::Exp::LiteralString(x) => Expr::String(x.0.clone().to_vec()), // TODO: why doesn't the MIR use Box<u8>?
            hir::Exp::Var(x) => match &x.0 {
                hir::Var::Name(x) => Expr::Var(self.get_var(&x.0)),
                hir::Var::Path { lhs, field } => {
                    let lhs = self.convert_exp(&lhs.0);
                    Expr::Index(x.as_ref().map(|_| IndexExpr {
                        base: Box::new(lhs),
                        index: Index::Name(field.0.into()),
                    }))
                }
                hir::Var::Index { lhs, idx } => {
                    let lhs = self.convert_exp(&lhs.0);
                    let idx = self.convert_exp(&idx.0);
                    Expr::Index(x.as_ref().map(|_| IndexExpr {
                        base: Box::new(lhs),
                        index: Index::Expr(Box::new(idx)),
                    }))
                }
            },
            x => todo!("{x:?}"),
        }
    }

    fn write_stat(&mut self, stat: &hir::Stat) {
        match stat {
            hir::Stat::FunctionCall(x) => {
                let lhs = self.convert_exp(&x.0.lhs.0);

                // TODO: this... this doesn't handle multires correctly, does it.
                let params = Multival::FixedList(
                    x.0.args.0.iter().map(|x| self.convert_exp(&x.0)).collect(),
                );

                self.cur_block
                    .push(Statement::Call(None, FunctionCall { base: lhs, params }));
            }
            x => todo!("{x:?}"),
        }
    }

    fn write_block_inner(&mut self, block_id: BasicBlockId, block: &hir::Block) {
        self.cur_block.id = block_id;
        for stat in &block.stats {
            self.write_stat(&stat.0);
        }
        self.basic_blocks.push(self.cur_block.finish_and_reset());
    }

    pub fn write_block(&mut self, block: &hir::Block) {
        self.write_block_inner(self.next_block, block);
    }

    pub fn finish(mut self) -> FunctionDef {
        self.basic_blocks.sort_by_key(|bb| bb.id.0);
        FunctionDef {
            debug_info: self.debug_info,
            num_upvars: self.num_upvars,
            num_params: self.num_params,
            variadic: self.variadic,
            blocks: self.basic_blocks,
        }
    }
}

#[cfg(test)]
mod test {
    use alloc::{boxed::Box, vec};
    use hashbrown::HashSet;
    use tsukiyotake_grammar::{s, synth};

    use crate::{bb, hir, mir::*, ssa_var};

    #[test]
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

        let mut mir_conv = MirConverter::new_at_root();
        mir_conv.write_block(&hir);
        let mir = mir_conv.finish();

        assert_eq!(
            mir,
            FunctionDef {
                debug_info: FunctionDebugInfo {
                    var_names_map: BTreeMap::from([
                        (ssa_var!(1), synth!("_ENV".into())),
                        (ssa_var!(2), synth!("...".into())),
                    ]),
                    function_canon_name: synth!("<root>".into())
                },
                num_upvars: 0,
                num_params: 0,
                variadic: true,
                blocks: vec![BasicBlock {
                    id: bb!(1),
                    stats: vec![Statement::Call(
                        None,
                        FunctionCall {
                            base: Expr::Index(s!(
                                IndexExpr {
                                    base: Box::new(Expr::Var(ssa_var!(1))),
                                    index: Index::Name("print".into()),
                                },
                                0..5
                            )),
                            params: Multival::FixedList(vec![Expr::String(b"Hello World".into())])
                        }
                    )],
                    term: None
                }],
            }
        )
    }
}
