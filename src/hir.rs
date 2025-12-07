use alloc::{boxed::Box, vec::Vec};
use tsukiyotake_grammar::ast::Spanned;

pub use crate::ast::{BinOp, UnOp};

pub type List<T> = Spanned<Vec<Spanned<T>>>;

#[derive(Clone, Debug, PartialEq)]
pub struct Block<'src> {
    pub stats: Vec<Spanned<Stat<'src>>>,
    pub retstat: Option<List<Exp<'src>>>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Stat<'src> {
    Empty,
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
    pub name: Spanned<&'src str>,
    pub attrib: Option<Spanned<&'src str>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FuncName<'src> {
    pub path: List<&'src str>,
    pub method: Option<Spanned<&'src str>>,
}

// Otherwise known as lvalue
#[derive(Clone, Debug, PartialEq)]
pub enum Var<'src> {
    Name(Spanned<&'src str>),
    Index {
        lhs: Box<Spanned<PrefixExp<'src>>>,
        idx: Box<Spanned<Exp<'src>>>,
    },
    Path {
        lhs: Box<Spanned<PrefixExp<'src>>>,
        member: Spanned<&'src str>,
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
    PrefixExp(Spanned<PrefixExp<'src>>),
    TableConstructor(List<Field<'src>>),
    BinExp {
        lhs: Box<Spanned<Exp<'src>>>,
        op: BinOp,
        rhs: Box<Spanned<Exp<'src>>>,
    },
    UnExp {
        op: UnOp,
        rhs: Box<Spanned<Exp<'src>>>,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub enum PrefixExp<'src> {
    Var(Spanned<Var<'src>>),
    FunctionCall(Spanned<FunctionCall<'src>>),
    Parens(Box<Spanned<Exp<'src>>>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct FunctionCall<'src> {
    pub lhs: Box<Spanned<PrefixExp<'src>>>,
    pub method: Option<Spanned<&'src str>>,
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
