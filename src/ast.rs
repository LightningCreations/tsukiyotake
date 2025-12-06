use alloc::{boxed::Box, vec::Vec};
use logos::Span;

pub type Spanned<T> = (T, Span);
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
    Function {
        name: Spanned<FuncName<'src>>,
        body: Spanned<FuncBody<'src>>,
    },
    LocalFunction {
        name: Spanned<&'src str>,
        body: Spanned<FuncBody<'src>>,
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
    pub args: Spanned<Args<'src>>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Args<'src> {
    List(List<Exp<'src>>),
    TableConstructor(List<Field<'src>>),
    String(Spanned<Box<[u8]>>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct FuncBody<'src> {
    pub params: List<&'src str>,
    pub varargs: Option<Spanned<()>>,
    pub block: Spanned<Block<'src>>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Field<'src> {
    Exp {
        field: Box<Spanned<Exp<'src>>>,
        val: Box<Spanned<Exp<'src>>>,
    },
    Named {
        field: Spanned<&'src str>,
        val: Box<Spanned<Exp<'src>>>,
    },
    Unnamed {
        val: Box<Spanned<Exp<'src>>>,
    },
}

impl<'src> Field<'src> {
    /// Convenience method for further stages
    pub fn val(&self) -> &Spanned<Exp<'src>> {
        match self {
            Field::Exp { val, .. } => val,
            Field::Named { val, .. } => val,
            Field::Unnamed { val } => val,
        }
    }
}

// most names are based on the related metatable function
#[derive(Clone, Debug, PartialEq, Hash, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Idiv,
    Pow,
    Mod,
    Band,
    Bxor,
    Bor,
    Shr,
    Shl,
    Concat,
    Lt,
    Le,
    Gt, // derived from Le
    Ge, // derived from Lt
    Eq,
    Neq, // derived from Eq
    And, // keyword/boolean
    Or,  // keyword/boolean
}

// most names are based on the related metatable function
#[derive(Clone, Debug, PartialEq, Hash, Eq)]
pub enum UnOp {
    Unm,
    Not, // keyword/boolean
    Len,
    Bnot,
}
