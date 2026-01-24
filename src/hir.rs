use core::sync::atomic::AtomicUsize;

use alloc::borrow::{Cow, ToOwned};
use alloc::rc::Rc;
use alloc::string::String;
use alloc::{boxed::Box, vec::Vec};
use alloc::{format, vec};
use hashbrown::HashSet;
use tsukiyotake_grammar::{s, synth};

use crate::ast;
pub use crate::ast::{BinOp, FuncName, List, Spanned, UnOp};

#[derive(Clone, Debug, PartialEq)]
pub struct Block<'src> {
    pub locals: HashSet<Cow<'src, str>>,
    pub import_locals: HashSet<Cow<'src, str>>,
    pub stats: Vec<Spanned<Stat<'src>>>,
    pub retstat: Option<List<Exp<'src>>>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Stat<'src> {
    Assign {
        vars: List<Var<'src>>,
        exps: List<Exp<'src>>,
    },
    FunctionCall {
        call: Spanned<FunctionCall<'src>>,
        target: Option<Cow<'src, str>>,
    },
    Label(Spanned<&'src str>),
    Break,
    Goto(Spanned<&'src str>),
    DoBlock(Box<Spanned<Block<'src>>>),
    While {
        cond: Spanned<Exp<'src>>,
        block: Box<Spanned<Block<'src>>>,
    },
    RepeatUntil {
        block: Box<Spanned<Block<'src>>>,
        cond: Spanned<Exp<'src>>,
    },
    If {
        cond_blocks: Vec<(Spanned<Exp<'src>>, Box<Spanned<Block<'src>>>)>,
        else_block: Option<Box<Spanned<Block<'src>>>>,
    },
    ForNumerical {
        var: Spanned<&'src str>,
        initial: Spanned<Exp<'src>>,
        limit: Spanned<Exp<'src>>,
        step: Option<Spanned<Exp<'src>>>,
        block: Box<Spanned<Block<'src>>>,
    },
    ForGeneric {
        names: List<&'src str>,
        exps: List<Exp<'src>>,
        block: Box<Spanned<Block<'src>>>,
    },
    Local {
        names: List<Cow<'src, str>>,
        exps: Option<List<Exp<'src>>>,
    },
}

// Otherwise known as lvalue
#[derive(Clone, Debug, PartialEq)]
pub enum Var<'src> {
    Name(Spanned<Cow<'src, str>>),
    Path {
        lhs: Box<Spanned<Exp<'src>>>,
        field: Spanned<Cow<'src, str>>,
    },
    Index {
        lhs: Box<Spanned<Exp<'src>>>,
        idx: Box<Spanned<Exp<'src>>>,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub enum Index<'src> {
    Name(Cow<'src, str>),
    Exp(Spanned<Exp<'src>>),
}

#[derive(Clone, Debug, PartialEq)]
pub enum Exp<'src> {
    Nil,
    False,
    True,
    NumeralInt(Spanned<i64>),
    NumeralFloat(Spanned<f64>),
    LiteralString(Spanned<Box<[u8]>>),
    VarArg,
    FunctionDef(Spanned<FuncBody<'src>>),
    TableConstructor {
        hash_part: Vec<(Index<'src>, Spanned<Exp<'src>>)>,
        array_part: Vec<Spanned<Exp<'src>>>,
    },
    BinExp {
        lhs: Box<Spanned<Exp<'src>>>,
        op: BinOp,
        rhs: Box<Spanned<Exp<'src>>>,
    },
    UnExp {
        op: UnOp,
        rhs: Box<Spanned<Exp<'src>>>,
    },
    Var(Spanned<Var<'src>>),
    CollapseMultival(Box<Spanned<Exp<'src>>>),
    FunctionCallResult(Cow<'src, str>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct FunctionCall<'src> {
    pub lhs: Box<Spanned<Exp<'src>>>,
    pub args: List<Exp<'src>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FuncBody<'src> {
    pub params: List<Cow<'src, str>>,
    pub varargs: Option<Spanned<()>>,
    pub block: Spanned<Block<'src>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Field<'src> {
    pub field: Box<Spanned<Exp<'src>>>,
    pub val: Box<Spanned<Exp<'src>>>,
}

#[derive(Clone)]
pub struct HirConversionContext<'src> {
    parent_locals: HashSet<Cow<'src, str>>,
    import_locals: HashSet<Cow<'src, str>>,
    locals: HashSet<Cow<'src, str>>,
    synthetic_counter: Rc<AtomicUsize>,
}

fn unbox<'src, T>(
    (result, mut new_stats): (T, Vec<Spanned<Stat<'src>>>),
    stats: &mut Vec<Spanned<Stat<'src>>>,
) -> T {
    stats.append(&mut new_stats);
    result
}

// path must be at least one element long
fn path_to_ast_var<'src>(path: &[Spanned<Cow<'src, str>>]) -> ast::Var<'src> {
    match path {
        [a] => ast::Var::Name(a.clone()),
        [a @ .., b] => {
            let lhs = path_to_ast_var(a);
            ast::Var::Path {
                lhs: Box::new(synth!(ast::PrefixExp::Var(synth!(lhs)))),
                member: b.clone(),
            }
        }
        [] => unreachable!(), // by contract
    }
}

impl<'src> HirConversionContext<'src> {
    pub fn new() -> Self {
        Self {
            parent_locals: HashSet::new(),
            import_locals: HashSet::new(),
            locals: HashSet::new(),
            synthetic_counter: Rc::new(AtomicUsize::new(0)),
        }
    }

    pub fn convert_block(mut self, ast: &ast::Block<'src>) -> Block<'src> {
        let mut stats = ast
            .stats
            .iter()
            .flat_map(|x| self.convert_stat(x.as_ref()))
            .collect();
        let retstat = ast.retstat.as_ref().map(|x| {
            x.as_ref().map(|x| {
                x.iter()
                    .map(|x| unbox(self.convert_exp(x.as_ref()), &mut stats))
                    .collect()
            })
        });
        Block {
            locals: self.locals,
            import_locals: self.import_locals,
            stats,
            retstat,
        }
    }

    pub fn convert_func_body(
        &mut self,
        ast: Spanned<&ast::FuncBody<'src>>,
        is_method: bool,
    ) -> Spanned<FuncBody<'src>> {
        let span = ast.1.clone();
        let mut params = ast.0.params.clone();

        if is_method {
            params.insert(0, synth!("self".into()));
        }

        let ctx = self.descend();
        let block = ast.0.block.as_ref().map(|x| ctx.convert_block(x));
        self.import_locals.extend(block.import_locals.clone());

        s!(
            FuncBody {
                params,
                varargs: ast.0.varargs.clone(),
                block
            },
            span
        )
    }

    fn fn_helper(
        &mut self,
        call: Spanned<&ast::FunctionCall<'src>>,
    ) -> (Spanned<FunctionCall<'src>>, Vec<Spanned<Stat<'src>>>) {
        // Needs to be handled specially; just about every possible function call scenario is different.
        let mut stats = vec![];
        let mut args = unbox(self.convert_args(call.args.as_ref()), &mut stats);
        let func_call = if let Some(method) = &call.method {
            // create local for object
            let local = self.new_local();
            let exps = Some(synth!(vec![unbox(
                self.convert_prefix_exp((*call.lhs).as_ref()),
                &mut stats
            )]));
            stats.push(synth!(Stat::Local {
                names: synth!(vec![synth!(local.clone())]),
                exps,
            }));
            // add self parameter
            args.insert(
                0,
                synth!(Exp::Var(synth!(Var::Name(synth!(local.clone(),))))),
            );
            // call the new function
            s!(
                FunctionCall {
                    lhs: Box::new(synth!(Exp::Var(synth!(Var::Path {
                        lhs: Box::new(synth!(Exp::Var(synth!(Var::Name(synth!(local)),)))),
                        field: method.clone(),
                    },)))),
                    args,
                },
                call.1
            )
        } else {
            s!(
                FunctionCall {
                    lhs: Box::new(unbox(
                        self.convert_prefix_exp((*call.lhs).as_ref()),
                        &mut stats
                    )),
                    args,
                },
                call.1
            )
        };
        (func_call, stats)
    }

    // ast::Stat::Empty is converted into None
    pub fn convert_stat(&mut self, ast: Spanned<&ast::Stat<'src>>) -> Vec<Spanned<Stat<'src>>> {
        let mut stats = vec![];
        match &*ast {
            ast::Stat::Empty => {}
            ast::Stat::Assign { vars, exps } => {
                let vars = vars.as_ref().map(|x| {
                    x.iter()
                        .map(|x| unbox(self.convert_var(x.as_ref()), &mut stats))
                        .collect()
                });
                let exps = exps.as_ref().map(|x| {
                    x.iter()
                        .map(|x| unbox(self.convert_exp(x.as_ref()), &mut stats))
                        .collect()
                });
                stats.push(s!(Stat::Assign { vars, exps }, ast.1));
            }
            ast::Stat::FunctionCall(call) => {
                let call = unbox(self.fn_helper(call.as_ref()), &mut stats);
                stats.push(s!(Stat::FunctionCall { call, target: None }, ast.1));
            }
            ast::Stat::Label(spanned) => todo!(),
            ast::Stat::Break => todo!(),
            ast::Stat::Goto(spanned) => todo!(),
            ast::Stat::DoBlock(spanned) => todo!(),
            ast::Stat::While { cond, block } => todo!(),
            ast::Stat::RepeatUntil { block, cond } => todo!(),
            ast::Stat::If {
                main,
                elseifs,
                else_block,
            } => {
                let mut conds = vec![main];
                conds.extend(elseifs.iter());
                let cond_blocks = conds
                    .into_iter()
                    .map(|(cond, block)| {
                        let cond = unbox(self.convert_exp(cond.as_ref()), &mut stats);
                        let ctx = self.descend();
                        let block = Box::new((**block).as_ref().map(|x| ctx.convert_block(x)));
                        self.import_locals.extend(block.import_locals.clone());
                        (cond, block)
                    })
                    .collect();

                let else_block = else_block.as_ref().map(|x| {
                    Box::new((**x).as_ref().map(|block| {
                        let ctx = self.descend();
                        let block = ctx.convert_block(block);
                        self.import_locals.extend(block.import_locals.clone());
                        block
                    }))
                });

                stats.push(s!(
                    Stat::If {
                        cond_blocks,
                        else_block,
                    },
                    ast.1
                ));
            }
            ast::Stat::ForNumerical {
                var,
                initial,
                limit,
                step,
                block,
            } => todo!(),
            ast::Stat::ForGeneric { names, exps, block } => todo!(),
            ast::Stat::Function { name, body } => {
                let mut path = name.path.clone();
                if let Some(method) = &name.method {
                    path.push(method.clone());
                    path.1.end = method.1.end;
                }
                let body = self.convert_func_body(body.as_ref(), name.method.is_some());
                let stat = s!(
                    Stat::Assign {
                        vars: synth!(vec![unbox(
                            self.convert_var(path.map(|x| path_to_ast_var(&x)).as_ref()),
                            &mut stats
                        )]),
                        exps: synth!(vec![synth!(Exp::FunctionDef(body))]),
                    },
                    ast.1
                );
                stats.push(stat);
            }
            ast::Stat::LocalFunction { name, body } => {
                let function = self.convert_func_body(body.as_ref(), false);
                stats.push(s!(
                    Stat::Local {
                        names: synth!(vec![name.clone()]),
                        exps: Some(synth!(vec![s!(Exp::FunctionDef(function), ast.1.clone())])),
                    },
                    ast.1
                ));
            }
            ast::Stat::Local { names, exps } => {
                let exps = exps.as_ref().map(|x| {
                    x.as_ref().map(|x| {
                        x.iter()
                            .map(|x| unbox(self.convert_exp(x.as_ref()), &mut stats))
                            .collect()
                    })
                });
                stats.push(s!(
                    Stat::Local {
                        names: names.as_ref().map(|x| x
                            .iter()
                            .map(|x| x.clone().map(|x| {
                                self.locals.insert(x.name.0.clone());
                                x.name.0
                            }))
                            .collect()),
                        exps,
                    },
                    ast.1
                ));
            }
            ast::Stat::Error => {
                unimplemented!("errors should be handled before attempting HIR translation")
            }
        }
        stats
    }

    pub fn convert_var(
        &mut self,
        ast: Spanned<&ast::Var<'src>>,
    ) -> (Spanned<Var<'src>>, Vec<Spanned<Stat<'src>>>) {
        let mut stats = vec![];
        let var = ast.map(|x| match x {
            ast::Var::Name(x) => {
                if self.in_scope(x) {
                    Var::Name(x.clone())
                } else {
                    Var::Path {
                        lhs: Box::new(synth!(Exp::Var(synth!(Var::Name(synth!("_ENV".into()),))))),
                        field: x.clone(),
                    }
                }
            }
            ast::Var::Index { lhs, idx } => Var::Index {
                lhs: Box::new(unbox(self.convert_prefix_exp((**lhs).as_ref()), &mut stats)),
                idx: Box::new(unbox(self.convert_exp((**idx).as_ref()), &mut stats)),
            },
            ast::Var::Path { lhs, member } => Var::Path {
                lhs: Box::new(unbox(self.convert_prefix_exp((**lhs).as_ref()), &mut stats)),
                field: member.clone(),
            },
        });
        (var, stats)
    }

    pub fn convert_exp(
        &mut self,
        ast: Spanned<&ast::Exp<'src>>,
    ) -> (Spanned<Exp<'src>>, Vec<Spanned<Stat<'src>>>) {
        let mut stats = vec![];
        let exp = ast.map(|x| match x {
            ast::Exp::Nil => Exp::Nil,
            ast::Exp::False => Exp::False,
            ast::Exp::True => Exp::True,
            ast::Exp::NumeralInt(x) => Exp::NumeralInt(x.clone()),
            ast::Exp::NumeralFloat(x) => Exp::NumeralFloat(x.clone()),
            ast::Exp::LiteralString(x) => Exp::LiteralString(x.clone()),
            ast::Exp::VarArg => Exp::VarArg,
            ast::Exp::FunctionDef(x) => Exp::FunctionDef(self.convert_func_body(x.as_ref(), false)),
            ast::Exp::PrefixExp(x) => unbox(self.convert_prefix_exp(x.as_ref()), &mut stats).0,
            ast::Exp::TableConstructor(x) => {
                let mut hash_part = Vec::new();
                let mut array_part = Vec::new();
                for field in &**x {
                    match &**field {
                        ast::Field::Exp { field, val } => {
                            hash_part.push((
                                Index::Exp(unbox(self.convert_exp((**field).as_ref()), &mut stats)),
                                unbox(self.convert_exp((**val).as_ref()), &mut stats),
                            ));
                        }
                        ast::Field::Named { field, val } => {
                            hash_part.push((
                                Index::Name(field.0.clone()),
                                unbox(self.convert_exp((**val).as_ref()), &mut stats),
                            ));
                        }
                        ast::Field::Unnamed { val } => {
                            array_part.push(unbox(self.convert_exp((**val).as_ref()), &mut stats));
                        }
                    }
                }
                Exp::TableConstructor {
                    hash_part,
                    array_part,
                }
            }
            ast::Exp::BinExp { lhs, op, rhs } => Exp::BinExp {
                lhs: Box::new(unbox(self.convert_exp((**lhs).as_ref()), &mut stats)),
                op: *op,
                rhs: Box::new(unbox(self.convert_exp((**rhs).as_ref()), &mut stats)),
            },
            ast::Exp::UnExp { op, rhs } => Exp::UnExp {
                op: *op,
                rhs: Box::new(unbox(self.convert_exp((**rhs).as_ref()), &mut stats)),
            },
            ast::Exp::Error => {
                unimplemented!("errors should be handled before attempting HIR translation")
            }
        });
        (exp, stats)
    }

    pub fn convert_prefix_exp(
        &mut self,
        ast: Spanned<&ast::PrefixExp<'src>>,
    ) -> (Spanned<Exp<'src>>, Vec<Spanned<Stat<'src>>>) {
        let mut stats = vec![];
        let exp = ast.map(|x| match x {
            ast::PrefixExp::Var(x) => Exp::Var(unbox(self.convert_var(x.as_ref()), &mut stats)),
            ast::PrefixExp::FunctionCall(x) => {
                let target = self.new_local();
                let call = s!(
                    Stat::FunctionCall {
                        call: unbox(self.fn_helper(x.as_ref()), &mut stats),
                        target: Some(target.clone()),
                    },
                    x.1.clone()
                );
                stats.push(call);
                Exp::FunctionCallResult(target)
            }
            ast::PrefixExp::Parens(x) => Exp::CollapseMultival(Box::new(unbox(
                self.convert_exp((**x).as_ref()),
                &mut stats,
            ))),
            ast::PrefixExp::Error => {
                unimplemented!("errors should be handled before attempting HIR translation")
            }
        });
        (exp, stats)
    }

    pub fn convert_args(
        &mut self,
        ast: Spanned<&ast::Args<'src>>,
    ) -> (List<Exp<'src>>, Vec<Spanned<Stat<'src>>>) {
        let mut stats = vec![];
        let args = match &*ast {
            ast::Args::List(list) => list.as_ref().map(|x| {
                x.iter()
                    .map(|x| unbox(self.convert_exp(x.as_ref()), &mut stats))
                    .collect()
            }),
            ast::Args::TableConstructor(x) => todo!(),
            ast::Args::String(x) => todo!(),
            ast::Args::Error => {
                unimplemented!("errors should be handled before attempting HIR translation")
            }
        };
        (args, stats)
    }

    // why does this take &mut self? because it will take steps to place things in scope if they're in the parent scope :)
    fn in_scope(&mut self, var: &str) -> bool {
        var == "_ENV"
            || self.locals.contains(var)
            || self.import_locals.contains(var)
            || if self.parent_locals.contains(var) {
                // side effects in a conditional? scandalous
                self.import_locals.insert(String::from(var).into());
                true
            } else {
                false
            }
    }

    fn descend(&self) -> Self {
        Self {
            parent_locals: self.parent_locals.union(&self.locals).cloned().collect(),
            import_locals: HashSet::new(),
            locals: self.locals.clone(),
            synthetic_counter: self.synthetic_counter.clone(),
        }
    }

    fn new_local(&mut self) -> Cow<'src, str> {
        let name = format!(
            "$synth_{}",
            self.synthetic_counter
                .fetch_add(1, core::sync::atomic::Ordering::Relaxed)
        );
        self.locals.insert(name.clone().into());
        name.into()
    }
}

#[cfg(test)]
mod test {
    use crate::{ast, hir::*};

    use alloc::{boxed::Box, vec};
    use tsukiyotake_grammar::{s, synth};

    #[test]
    fn hello_world() {
        let ast = ast::Block {
            stats: vec![s!(
                ast::Stat::FunctionCall(s!(
                    ast::FunctionCall {
                        lhs: Box::new(s!(
                            ast::PrefixExp::Var(s!(ast::Var::Name(s!("print".into(), 0..5)), 0..5)),
                            0..5
                        )),
                        method: None,
                        args: s!(
                            ast::Args::List(s!(
                                vec![s!(
                                    ast::Exp::LiteralString(s!(
                                        Box::new(*br#"Hello World"#),
                                        6..19
                                    )),
                                    6..19
                                )],
                                6..19
                            )),
                            5..20
                        )
                    },
                    0..20
                )),
                0..20
            )],
            retstat: None,
        };

        let conv = HirConversionContext::new();
        let hir = conv.convert_block(&ast);

        assert_eq!(
            hir,
            Block {
                locals: HashSet::new(),
                import_locals: HashSet::new(),
                stats: vec![s!(
                    Stat::FunctionCall {
                        call: s!(
                            FunctionCall {
                                lhs: Box::new(s!(
                                    Exp::Var(s!(
                                        Var::Path {
                                            lhs: Box::new(synth!(Exp::Var(synth!(Var::Name(
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
                                        Exp::LiteralString(s!(Box::new(*br#"Hello World"#), 6..19)),
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
                retstat: None
            }
        );
    }
}
