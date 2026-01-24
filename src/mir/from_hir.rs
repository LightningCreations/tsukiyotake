use alloc::borrow::Cow;
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
pub struct MirConverter<'src> {
    debug_info: FunctionDebugInfo,
    vars_by_name: HashMap<Cow<'src, str>, SsaVarId>,
    num_upvars: u32,
    num_params: u32,
    variadic: bool,
    basic_blocks: Vec<BasicBlock>,
    cur_block: UnbuiltBasicBlock,
    next_var: SsaVarId,
    next_block: BasicBlockId,
}

impl<'src> MirConverter<'src> {
    pub fn new(
        name: Spanned<Cow<'src, str>>,
        upvars: &[Spanned<Cow<'src, str>>],
        params: &[Spanned<Cow<'src, str>>],
        variadic: bool,
    ) -> Self {
        let mut result = Self {
            debug_info: FunctionDebugInfo {
                var_names_map: BTreeMap::new(),
                function_canon_name: name.map(|x| x.into()),
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

    fn add_var(&mut self, name: Spanned<Cow<'src, str>>) -> SsaVarId {
        let var = self.next_var;
        self.vars_by_name.insert(name.0.clone(), var);
        self.debug_info
            .var_names_map
            .insert(var, name.map(|x| x.into()));
        self.next_var = var.next();
        var
    }

    pub fn new_at_root() -> Self {
        // Handles the root chunk, as it has very particular settings
        Self::new(synth!("<root>".into()), &[], &[], true)
    }

    fn get_var(&mut self, name: &'src str) -> SsaVarId {
        if let Some(x) = self.vars_by_name.get(name) {
            *x
        } else {
            self.add_var(synth!(name.into())) // TODO: add the variable with the correct span
        }
    }

    fn convert_exp(&mut self, exp: Spanned<&'src hir::Exp<'src>>) -> Expr {
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
            hir::Exp::FunctionDef(x) => {
                let mut new_conv = MirConverter::new(
                    synth!("<TODO: DETERMINE CANONICAL NAME>".into()),
                    &x.block
                        .import_locals
                        .iter()
                        .cloned()
                        .map(|x| synth!(x))
                        .collect::<Vec<_>>(),
                    &x.params,
                    x.varargs.is_some(),
                );
                new_conv.write_block(&x.block);
                let function = new_conv.finish();
                Expr::Closure(ClosureDef {
                    captures: x
                        .block
                        .import_locals
                        .iter()
                        .map(|x| self.get_var(&x))
                        .collect(),
                    function,
                })
            }
            hir::Exp::FunctionCallResult(x) => Expr::Extract(
                Box::new(Multival::Var {
                    var: self.get_var(&x),
                    count: None,
                }),
                0,
            ),
            x => todo!("{x:?}"),
        }
    }

    fn convert_exp_multival(&mut self, exp: Spanned<&'src hir::Exp<'src>>) -> Multival {
        match exp.0 {
            hir::Exp::FunctionCallResult(x) => Multival::Var {
                var: self.get_var(&x),
                count: None,
            },
            _ => Multival::FixedList(vec![self.convert_exp(exp)]),
        }
    }

    fn convert_list(&mut self, exps: &'src [Spanned<hir::Exp<'src>>]) -> Multival {
        let Some((last, first)) = exps.split_last() else {
            return Multival::Empty;
        };
        let first = first.iter().map(|x| self.convert_exp(x.as_ref())).collect();
        let last = self.convert_exp_multival(last.as_ref());
        Multival::Concat(vec![Multival::FixedList(first), last])
    }

    /// Returns true if a terminator was written
    fn write_stat(
        &mut self,
        stat: Spanned<&'src hir::Stat<'src>>,
        break_block_id: Option<BasicBlockId>,
        continue_block_id: Option<BasicBlockId>,
    ) -> bool {
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
                            self.add_var(x.as_ref().map(|x| x.clone()))
                        } else {
                            self.add_var(synth!(format!("<temporary assignment index {i}>").into()))
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
            hir::Stat::FunctionCall { call, target } => {
                let lhs = self.convert_exp((*call.lhs).as_ref());
                let params = self.convert_list(&call.args);
                let target = target.as_ref().map(|x| self.add_var(synth!(x.clone())));

                self.cur_block.push(s!(
                    Statement::Call(target, FunctionCall { base: lhs, params }),
                    call.1.clone(),
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
                self.next_block = cur_block_id;
                for (id, block) in code_block_ids.into_iter().zip(code_blocks) {
                    self.write_block_inner(
                        id,
                        block,
                        Some(final_terminator.clone()),
                        break_block_id,
                        continue_block_id,
                    );
                }
                self.cur_block.id = cur_block_id;
            }
            hir::Stat::Local { names, exps } => {
                for name in &names.0 {
                    self.add_var(name.clone());
                }
                if let Some(exps) = exps {
                    let result = self.convert_list(exps);
                    let statement = s!(
                        Statement::Multideclare(
                            names.0.iter().map(|x| self.get_var(&x.0)).collect(),
                            result
                        ),
                        stat.1
                    );
                    self.cur_block.push(statement);
                }
            }
            hir::Stat::While { cond, block } => {
                // This is a bit similar to if-else blocks in terms of how we handle it, the main difference being that in this we loop. Easy, right?

                // Step 1: figure out what block ids we care about
                let continue_block_id = self.next_block.next(); // contains the loop branch itself
                let break_block_id = continue_block_id.next(); // contains a jump to the end; we have to fill this out last.
                let loop_body_id = break_block_id.next(); // contains the start of the loop body

                // Step 2: write the jump to the condition
                self.basic_blocks
                    .push(
                        self.cur_block
                            .finish_and_reset(Some(synth!(Terminator::Jump(JumpTarget {
                                targ_bb: continue_block_id,
                                remaps: vec![]
                            })))),
                    );

                // Step 3: write the condition block
                self.cur_block.id = continue_block_id;
                let cond = self.convert_exp(cond.as_ref());
                self.basic_blocks
                    .push(
                        self.cur_block
                            .finish_and_reset(Some(synth!(Terminator::Branch(
                                cond,
                                JumpTarget {
                                    targ_bb: loop_body_id,
                                    remaps: vec![]
                                },
                                JumpTarget {
                                    targ_bb: break_block_id,
                                    remaps: vec![]
                                }
                            )))),
                    );

                // Step 4: write the loop body itself
                self.next_block = loop_body_id;
                self.write_block_inner(
                    loop_body_id,
                    block,
                    Some(synth!(Terminator::Jump(JumpTarget {
                        targ_bb: continue_block_id,
                        remaps: vec![]
                    }))),
                    Some(break_block_id),
                    Some(continue_block_id),
                );

                // Step 5: write the final block
                self.cur_block.id = break_block_id;
                self.basic_blocks
                    .push(
                        self.cur_block
                            .finish_and_reset(Some(synth!(Terminator::Jump(JumpTarget {
                                targ_bb: self.next_block,
                                remaps: vec![]
                            })))),
                    );
                self.cur_block.id = self.next_block;
            }
            hir::Stat::Break => {
                if let Some(break_block_id) = break_block_id {
                    self.basic_blocks
                        .push(self.cur_block.finish_and_reset(Some(s!(
                            Terminator::Jump(JumpTarget {
                                targ_bb: break_block_id,
                                remaps: vec![],
                            }),
                            stat.1
                        ))));
                } else {
                    self.basic_blocks
                        .push(self.cur_block.finish_and_reset(Some(s!(
                            Terminator::DeferError(synth!("break executed outside of loop".into())),
                            stat.1
                        ))));
                }
                return true;
            }
            x => todo!("{x:?}"),
        }
        false
    }

    fn write_block_inner(
        &mut self,
        block_id: BasicBlockId,
        block: &'src hir::Block<'src>,
        terminator: Option<Spanned<Terminator>>,
        break_block_id: Option<BasicBlockId>,
        continue_block_id: Option<BasicBlockId>,
    ) {
        self.cur_block.id = block_id;
        for stat in &block.stats {
            if self.write_stat(stat.as_ref(), break_block_id, continue_block_id) {
                self.next_block = self.next_block.next();
                return; // Terminator has been created early; bail
            }
        }
        if let Some(retstat) = &block.retstat {
            let terminator = retstat
                .as_ref()
                .map(|x| Terminator::Return(self.convert_list(&x)));
            self.basic_blocks
                .push(self.cur_block.finish_and_reset(Some(terminator)));
        } else {
            self.basic_blocks
                .push(self.cur_block.finish_and_reset(terminator));
        }
        self.next_block = self.next_block.next();
    }

    pub fn write_block(&mut self, block: &'src hir::Block<'src>) {
        self.write_block_inner(self.next_block, block, None, None, None);
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
                hir::Stat::FunctionCall {
                    call: s!(
                        hir::FunctionCall {
                            lhs: Box::new(s!(
                                hir::Exp::Var(s!(
                                    hir::Var::Path {
                                        lhs: Box::new(synth!(hir::Exp::Var(synth!(
                                            hir::Var::Name(synth!("_ENV".into()))
                                        )))),
                                        field: s!("print".into(), 0..5)
                                    },
                                    0..5,
                                )),
                                0..5,
                            )),
                            args: s!(
                                vec![s!(
                                    hir::Exp::LiteralString(s!(
                                        Box::new(*br#"Hello World"#),
                                        6..19
                                    )),
                                    6..19,
                                )],
                                6..19,
                            )
                        },
                        0..20,
                    ),
                    target: None
                },
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
