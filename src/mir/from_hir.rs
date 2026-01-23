use alloc::format;
use alloc::vec;
use hashbrown::HashMap;
use tsukiyotake_grammar::s;
use tsukiyotake_grammar::synth;

use crate::{bb, hir, mir::*};

#[derive(Debug)]
struct UnbuiltBasicBlock {
    id: BasicBlockId,
    stats: Vec<Spanned<Statement>>,
}

impl UnbuiltBasicBlock {
    fn new() -> Self {
        Self {
            id: BasicBlockId::new_unchecked(NonZeroU32::new(1).unwrap()),
            stats: Vec::new(),
        }
    }

    fn finish_and_reset(&mut self, terminator: Option<Spanned<Terminator>>) -> BasicBlock {
        BasicBlock {
            id: self.id,
            stats: core::mem::take(&mut self.stats),
            term: terminator.unwrap_or(synth!(Terminator::Return(Multival::Empty))), // TODO: Make this a real span
        }
    }

    fn push(&mut self, stat: Spanned<Statement>) {
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

        result.add_var(synth!("_ENV".into()));
        for param in params {
            result.add_var(param.clone());
        }
        if variadic {
            result.add_var(synth!("...".into()));
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
        Self::new(synth!("<root>".into()), &[], &[], true)
    }

    fn get_var(&mut self, name: &str) -> SsaVarId {
        if let Some(x) = self.vars_by_name.get(name) {
            *x
        } else {
            todo!()
        }
    }

    fn convert_exp(&mut self, exp: Spanned<&hir::Exp>) -> Expr {
        match exp.0 {
            hir::Exp::Nil => Expr::Nil,
            hir::Exp::False => Expr::Boolean(false),
            hir::Exp::True => Expr::Boolean(true),
            hir::Exp::NumeralInt(x) => Expr::Integer(x.0),
            hir::Exp::NumeralFloat(x) => Expr::Float(x.to_bits()),
            hir::Exp::LiteralString(x) => Expr::String(x.0.clone().to_vec()),
            hir::Exp::Var(x) => match &**x {
                hir::Var::Name(x) => Expr::Var(self.get_var(&**x)),
                hir::Var::Path { lhs, field } => {
                    let lhs = self.convert_exp((**lhs).as_ref());
                    Expr::Index(x.as_ref().map(|_| IndexExpr {
                        base: Box::new(lhs),
                        index: Index::Name(field.0.clone().into_owned()),
                    }))
                }
                hir::Var::Index { lhs, idx } => {
                    let lhs = self.convert_exp((**lhs).as_ref());
                    let idx = self.convert_exp((**idx).as_ref());
                    Expr::Index(x.as_ref().map(|_| IndexExpr {
                        base: Box::new(lhs),
                        index: Index::Expr(Box::new(idx)),
                    }))
                }
            },
            hir::Exp::TableConstructor {
                hash_part,
                array_part,
            } => Expr::Table(TableConstructor {
                hash_part: hash_part
                    .iter()
                    .map(|(index, exp)| {
                        (
                            match index {
                                hir::Index::Name(name) => Index::Name(name.clone().into()),
                                hir::Index::Exp(exp) => {
                                    Index::Expr(Box::new(self.convert_exp(exp.as_ref())))
                                }
                            },
                            self.convert_exp(exp.as_ref()),
                        )
                    })
                    .collect(),
                array_part: Box::new(self.convert_list(array_part)),
            }),
            hir::Exp::BinExp { lhs, op, rhs } => Expr::BinaryOp(s!(
                BinaryExpr {
                    op: *op,
                    left: Box::new(self.convert_exp((**lhs).as_ref())),
                    right: Box::new(self.convert_exp((**rhs).as_ref())),
                },
                exp.1,
            )),
            hir::Exp::UnExp { op, rhs } => Expr::UnaryOp(s!(
                UnaryExpr {
                    op: *op,
                    expr: Box::new(self.convert_exp((**rhs).as_ref()))
                },
                exp.1
            )),
            hir::Exp::CollapseMultival(x) => {
                Expr::Extract(Box::new(self.convert_exp_multival((**x).as_ref())), 0)
            }
            x => todo!("{x:?}"),
        }
    }

    fn convert_exp_multival(&mut self, exp: Spanned<&hir::Exp>) -> Multival {
        match exp.0 {
            hir::Exp::FunctionCall(_) => todo!(),
            _ => Multival::FixedList(vec![self.convert_exp(exp)]),
        }
    }

    fn convert_list(&mut self, exps: &[Spanned<hir::Exp>]) -> Multival {
        let Some((last, first)) = exps.split_last() else {
            return Multival::Empty;
        };
        let first = first.iter().map(|x| self.convert_exp(x.as_ref())).collect();
        let last = self.convert_exp_multival(last.as_ref());
        Multival::Concat(vec![Multival::FixedList(first), last])
    }

    fn write_stat(&mut self, stat: Spanned<&hir::Stat>) {
        match stat.0 {
            hir::Stat::Assign { vars, exps } => {
                // Figure out what assignments we need to make first
                let targets: Vec<_> = vars
                    .iter()
                    .map(|x| match &x.0 {
                        hir::Var::Name(_) => None,
                        hir::Var::Index { lhs, idx } => Some((
                            self.convert_exp((**lhs).as_ref()),
                            Index::Expr(Box::new(self.convert_exp((**idx).as_ref()))),
                            x.1.clone(),
                        )),
                        hir::Var::Path { lhs, field } => Some((
                            self.convert_exp((**lhs).as_ref()),
                            Index::Name(field.0.clone().into_owned()),
                            x.1.clone(),
                        )),
                    })
                    .collect();
                // Then, generate the result
                let result = self.convert_list(exps);
                // Then, generate the list of (temporary or otherwise) variables assigned to
                let vars: Vec<_> = vars
                    .iter()
                    .enumerate()
                    .map(|(i, x)| {
                        if let hir::Var::Name(x) = &**x {
                            self.add_var(x.as_ref().map(|x| x.clone().into_owned()))
                        } else {
                            self.add_var(synth!(format!("<temporary assignment index {i}>")))
                        }
                    })
                    .collect();
                // Then, generate the actual assignment statement
                self.cur_block
                    .push(s!(Statement::Multideclare(vars.clone(), result), stat.1,));
                // Finally, write any temporary variables to their correct targets
                for (i, target) in targets.into_iter().enumerate() {
                    if let Some((table, index, span)) = target {
                        self.cur_block.push(s!(
                            Statement::WriteIndex {
                                table,
                                index,
                                value: Expr::Var(vars[i]),
                            },
                            span,
                        ));
                    }
                }
            }
            hir::Stat::FunctionCall(x) => {
                let lhs = self.convert_exp((*x.lhs).as_ref());
                let params = self.convert_list(&x.args);

                self.cur_block.push(s!(
                    Statement::Call(None, FunctionCall { base: lhs, params }),
                    x.1.clone(),
                ));
            }
            hir::Stat::If {
                cond_blocks,
                else_block,
            } => {
                // Alright. This is *not* the structure we want, so we're gonna fix it.
                // What we will generate is a set of block terminators, then the corresponding blocks.
                // Each block terminator will be an if-else in effect.
                // As an example, this Lua code:
                //
                //     if a then
                //         block1
                //     elseif b then
                //         block2
                //     else
                //         block3
                //     end
                //
                // turns into this code, roughly:
                //
                //     @1: {
                //         branch a @2 else @3
                //     }
                //     @2: {
                //         block1
                //         jump @6
                //     }
                //     @3: {
                //         branch b @4 else @5
                //     }
                //     @4: {
                //         block2
                //         jump @6
                //     }
                //     @5: {
                //         block3
                //         jump @6
                //     }
                //
                // and then code continues at @6.
                // This more-or-less mimics an if-else tree rather than an if-elseif-else fan,
                // and while we could rewrite into that form, I don't want to do that much work, so I won't.
                // Instead, we'll do this in stages.

                // First, we'll generate the block terminators for the "scaffolding" of the fan, i.e. everything but the code inside the if-else blocks.
                let mut cur_block_id = self.next_block;
                let mut terminators = Vec::new();
                let mut code_block_ids = Vec::new();
                for (exp, _) in cond_blocks {
                    let code_block_id = cur_block_id.next();
                    let next_block_id = code_block_id.next();
                    terminators.push((
                        cur_block_id,
                        s!(
                            Terminator::Branch(
                                self.convert_exp(exp.as_ref()),
                                JumpTarget {
                                    targ_bb: code_block_id,
                                    remaps: Vec::new(),
                                },
                                JumpTarget {
                                    targ_bb: next_block_id,
                                    remaps: Vec::new(),
                                },
                            ),
                            exp.1.clone()
                        ),
                    ));
                    code_block_ids.push(code_block_id);
                    cur_block_id = next_block_id;
                }
                let mut code_blocks = cond_blocks.iter().map(|(_, x)| &**x).collect::<Vec<_>>();
                if let Some(else_block) = else_block {
                    code_block_ids.push(cur_block_id);
                    cur_block_id = cur_block_id.next();
                    code_blocks.push(else_block);
                }
                let final_terminator = synth!(Terminator::Jump(JumpTarget {
                    targ_bb: cur_block_id,
                    remaps: Vec::new()
                }));
                for (id, term) in terminators {
                    self.cur_block.id = id;
                    self.basic_blocks
                        .push(self.cur_block.finish_and_reset(Some(term)));
                }
                for (id, block) in code_block_ids.into_iter().zip(code_blocks) {
                    self.write_block_inner(id, block, Some(final_terminator.clone()));
                }
                self.cur_block.id = cur_block_id;
            }
            x => todo!("{x:?}"),
        }
    }

    fn write_block_inner(
        &mut self,
        block_id: BasicBlockId,
        block: &hir::Block,
        terminator: Option<Spanned<Terminator>>,
    ) {
        self.cur_block.id = block_id;
        for stat in &block.stats {
            self.write_stat(stat.as_ref());
        }
        if block.retstat.is_none() {
            self.basic_blocks
                .push(self.cur_block.finish_and_reset(terminator));
        } else {
            todo!()
        }
    }

    pub fn write_block(&mut self, block: &hir::Block) {
        self.write_block_inner(self.next_block, block, None);
    }

    pub fn finish(mut self) -> FunctionDef {
        self.basic_blocks.sort_by_key(|bb| bb.id.0);
        FunctionDef {
            debug_info: self.debug_info,
            num_upvars: self.num_upvars,
            num_params: self.num_params,
            num_ssa: self.next_var.0.get(),
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
                                    field: s!("print".into(), 0..5)
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
                num_ssa: 2,
                variadic: true,
                blocks: vec![BasicBlock {
                    id: bb!(1),
                    stats: vec![s!(
                        Statement::Call(
                            None,
                            FunctionCall {
                                base: Expr::Index(s!(
                                    IndexExpr {
                                        base: Box::new(Expr::Var(ssa_var!(1))),
                                        index: Index::Name("print".into()),
                                    },
                                    0..5
                                )),
                                params: Multival::FixedList(vec![Expr::String(
                                    b"Hello World".into()
                                )])
                            }
                        ),
                        0..20
                    )],
                    term: synth!(Terminator::Return(Multival::Empty)),
                }],
            }
        )
    }
}
