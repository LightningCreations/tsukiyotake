use core::sync::atomic::AtomicUsize;

use alloc::borrow::ToOwned;
use alloc::rc::Rc;
use alloc::string::String;
use alloc::{boxed::Box, vec::Vec};
use alloc::{format, vec};
use hashbrown::HashSet;

use crate::ast;
pub use crate::ast::{BinOp, FuncName, List, Spanned, UnOp};

#[derive(Clone, Debug, PartialEq)]
pub struct Block<'src> {
    pub locals: HashSet<String>,
    pub import_locals: HashSet<String>,
    pub stats: Vec<Spanned<Stat<'src>>>,
    pub retstat: Option<List<Exp<'src>>>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Stat<'src> {
    Assign {
        vars: List<Var<'src>>,
        exps: List<Exp<'src>>,
    },
    FunctionCall(Spanned<FunctionCall<'src>>),
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
        main: (Spanned<Exp<'src>>, Box<Spanned<Block<'src>>>),
        elseifs: Vec<(Spanned<Exp<'src>>, Box<Spanned<Block<'src>>>)>,
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
        names: List<AttName<'src>>,
        exps: Option<List<Exp<'src>>>,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct AttName<'src> {
    pub name: Spanned<String>,
    pub attrib: Option<Spanned<&'src str>>,
}

// Otherwise known as lvalue
#[derive(Clone, Debug, PartialEq)]
pub enum Var<'src> {
    Name(Spanned<String>),
    Path {
        lhs: Box<Spanned<Exp<'src>>>,
        field: Spanned<&'src str>,
    },
    Index {
        lhs: Box<Spanned<Exp<'src>>>,
        idx: Box<Spanned<Exp<'src>>>,
    },
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
    TableConstructor,
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
    FunctionCall(Spanned<FunctionCall<'src>>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct FunctionCall<'src> {
    pub lhs: Box<Spanned<Exp<'src>>>,
    pub args: List<Exp<'src>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FuncBody<'src> {
    pub params: List<&'src str>,
    pub varargs: Option<Spanned<()>>,
    pub block: Spanned<Block<'src>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Field<'src> {
    pub field: Box<Spanned<Exp<'src>>>,
    pub val: Box<Spanned<Exp<'src>>>,
}

#[derive(Clone)]
pub struct HirConversionContext {
    parent_locals: HashSet<String>,
    import_locals: HashSet<String>,
    locals: HashSet<String>,
    synthetic_counter: Rc<AtomicUsize>,
}

impl HirConversionContext {
    pub fn new() -> Self {
        Self {
            parent_locals: HashSet::new(),
            import_locals: HashSet::new(),
            locals: HashSet::new(),
            synthetic_counter: Rc::new(AtomicUsize::new(0)),
        }
    }

    pub fn convert_block<'src>(mut self, ast: &ast::Block<'src>) -> Block<'src> {
        let stats = ast
            .stats
            .iter()
            .flat_map(|x| self.convert_stat(x.as_ref()))
            .collect();
        let retstat = ast.retstat.as_ref().map(|x| {
            x.as_ref()
                .map(|x| x.iter().map(|x| self.convert_exp(x.as_ref())).collect())
        });
        Block {
            locals: self.locals,
            import_locals: self.import_locals,
            stats,
            retstat,
        }
    }

    // ast::Stat::Empty is converted into None
    pub fn convert_stat<'src>(
        &mut self,
        ast: Spanned<&ast::Stat<'src>>,
    ) -> Vec<Spanned<Stat<'src>>> {
        match ast.0 {
            ast::Stat::Empty => vec![],
            ast::Stat::Assign { vars, exps } => todo!(),
            ast::Stat::FunctionCall(call) => {
                let call = call.as_ref();
                // Needs to be handled specially; just about every possible function call scenario is different.
                let mut stats = vec![];
                let mut args = self.convert_args(call.0.args.as_ref());
                if let Some(method) = &call.0.method {
                    // create local for object
                    let local = self.new_local();
                    stats.push(Spanned::synth(Stat::Local {
                        names: Spanned::synth(vec![Spanned::synth(AttName {
                            name: Spanned::synth(local.clone()),
                            attrib: None,
                        })]),
                        exps: Some(Spanned::synth(vec![
                            self.convert_prefix_exp((*call.0.lhs).as_ref()),
                        ])),
                    }));
                    // add self parameter
                    args.0.insert(
                        0,
                        Spanned::synth(Exp::Var(Spanned::synth(Var::Name(Spanned::synth(
                            local.clone(),
                        ))))),
                    );
                    // call the new function
                    stats.push(Spanned(
                        Stat::FunctionCall(Spanned(
                            FunctionCall {
                                lhs: Box::new(Spanned::synth(Exp::Var(Spanned::synth(
                                    Var::Path {
                                        lhs: Box::new(Spanned::synth(Exp::Var(Spanned::synth(
                                            Var::Name(Spanned::synth(local)),
                                        )))),
                                        field: method.clone(),
                                    },
                                )))),
                                args,
                            },
                            call.1,
                        )),
                        ast.1,
                    ));
                } else {
                    stats.push(Spanned(
                        Stat::FunctionCall(Spanned(
                            FunctionCall {
                                lhs: Box::new(self.convert_prefix_exp((*call.0.lhs).as_ref())),
                                args,
                            },
                            call.1,
                        )),
                        ast.1,
                    ));
                }
                stats
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
            } => todo!(),
            ast::Stat::ForNumerical {
                var,
                initial,
                limit,
                step,
                block,
            } => todo!(),
            ast::Stat::ForGeneric { names, exps, block } => todo!(),
            ast::Stat::Function { name, body } => todo!(),
            ast::Stat::LocalFunction { name, body } => todo!(),
            ast::Stat::Local { names, exps } => todo!(),
        }
    }

    pub fn convert_var<'src>(&mut self, ast: Spanned<&ast::Var<'src>>) -> Spanned<Var<'src>> {
        ast.map(|x| match x {
            ast::Var::Name(x) => {
                if self.in_scope(x.0) {
                    Var::Name(x.as_ref().map(|x| (*x).into()))
                } else {
                    Var::Path {
                        lhs: Box::new(Spanned::synth(Exp::Var(Spanned::synth(Var::Name(
                            Spanned::synth("_ENV".into()),
                        ))))),
                        field: x.clone(),
                    }
                }
            }
            ast::Var::Index { lhs, idx } => todo!(),
            ast::Var::Path { lhs, member } => todo!(),
        })
    }

    pub fn convert_exp<'src>(&mut self, ast: Spanned<&ast::Exp<'src>>) -> Spanned<Exp<'src>> {
        ast.map(|x| match x {
            ast::Exp::Nil => todo!(),
            ast::Exp::False => todo!(),
            ast::Exp::True => todo!(),
            ast::Exp::NumeralInt(spanned) => todo!(),
            ast::Exp::NumeralFloat(spanned) => todo!(),
            ast::Exp::LiteralString(x) => Exp::LiteralString(x.clone()),
            ast::Exp::VarArg => todo!(),
            ast::Exp::FunctionDef(spanned) => todo!(),
            ast::Exp::PrefixExp(spanned) => todo!(),
            ast::Exp::TableConstructor(spanned) => todo!(),
            ast::Exp::BinExp { lhs, op, rhs } => todo!(),
            ast::Exp::UnExp { op, rhs } => todo!(),
        })
    }

    pub fn convert_prefix_exp<'src>(
        &mut self,
        ast: Spanned<&ast::PrefixExp<'src>>,
    ) -> Spanned<Exp<'src>> {
        ast.map(|x| match x {
            ast::PrefixExp::Var(x) => Exp::Var(self.convert_var(x.as_ref())),
            ast::PrefixExp::FunctionCall(spanned) => todo!(),
            ast::PrefixExp::Parens(spanned) => todo!(),
        })
    }

    pub fn convert_args<'src>(&mut self, ast: Spanned<&ast::Args<'src>>) -> List<Exp<'src>> {
        match ast.0 {
            ast::Args::List(list) => list
                .as_ref()
                .map(|x| x.iter().map(|x| self.convert_exp(x.as_ref())).collect()),
            ast::Args::TableConstructor(x) => todo!(),
            ast::Args::String(x) => todo!(),
        }
    }

    // why does this take &mut self? because it will take steps to place things in scope if they're in the parent scope :)
    fn in_scope(&mut self, var: &str) -> bool {
        var == "_ENV"
            || self.locals.contains(var)
            || self.import_locals.contains(var)
            || if self.parent_locals.contains(var) {
                // side effects in a conditional? scandalous
                self.import_locals.insert(var.to_owned());
                true
            } else {
                false
            }
    }

    fn descend(&self) -> Self {
        Self {
            parent_locals: self
                .parent_locals
                .union(&self.locals)
                .map(|x| x.clone())
                .collect(),
            import_locals: HashSet::new(),
            locals: self.locals.clone(),
            synthetic_counter: self.synthetic_counter.clone(),
        }
    }

    fn new_local(&mut self) -> String {
        let name = format!(
            "$synth_{}",
            self.synthetic_counter
                .fetch_add(1, core::sync::atomic::Ordering::Relaxed)
        );
        self.locals.insert(name.clone());
        name
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
                            ast::PrefixExp::Var(s!(ast::Var::Name(s!("print", 0..5)), 0..5)),
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
                    Stat::FunctionCall(s!(
                        FunctionCall {
                            lhs: Box::new(s!(
                                Exp::Var(s!(
                                    Var::Path {
                                        lhs: Box::new(synth!(Exp::Var(synth!(Var::Name(synth!(
                                            "_ENV".into()
                                        )))))),
                                        field: s!("print", 0..5)
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
                    )),
                    0..20,
                )],
                retstat: None
            }
        );
    }
}
