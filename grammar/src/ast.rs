use core::fmt;
use std::{
    fmt::Write,
    ops::{Deref, DerefMut},
};

use alloc::{boxed::Box, vec::Vec};
use logos::Span;

// TODO: redo this so it is Copy
#[derive(Clone, Debug, PartialEq, Hash, Eq)]
pub struct Spanned<T>(pub T, pub Span);

impl<T> Deref for Spanned<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.0
    }
}

impl<T> DerefMut for Spanned<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.0
    }
}

impl<T: fmt::Display> fmt::Display for Spanned<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[macro_export]
macro_rules! s {
    ($x:expr, $y:expr $(,)?) => {
        $crate::ast::Spanned($x, $y)
    };
}

#[macro_export]
macro_rules! synth {
    ($x:expr $(,)?) => {
        $crate::ast::Spanned::synth($x)
    };
}

impl<T> Spanned<T> {
    pub fn as_ref(&self) -> Spanned<&T> {
        Spanned(&self.0, self.1.clone()) // practically free clone
    }

    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> Spanned<U> {
        Spanned(f(self.0), self.1)
    }

    pub fn try_map<E, U>(self, f: impl FnOnce(T) -> Result<U, E>) -> Result<Spanned<U>, E> {
        Ok(Spanned(f(self.0)?, self.1))
    }

    pub fn synth(inner: T) -> Self {
        Self(inner, 0..0)
    }
}

impl<T> Spanned<Option<T>> {
    pub fn transpose(self) -> Option<Spanned<T>> {
        match self.0 {
            None => None,
            Some(x) => Some(Spanned(x, self.1)),
        }
    }
}

pub type List<T> = Spanned<Vec<Spanned<T>>>;

#[derive(Clone, Debug, PartialEq)]
pub struct Block<'src> {
    pub stats: Vec<Spanned<Stat<'src>>>,
    pub retstat: Option<List<Exp<'src>>>,
}

impl Block<'_> {
    pub fn pretty_print(&self, f: &mut fmt::Formatter, pad: &String) -> fmt::Result {
        for stat in &self.stats {
            write!(f, "{pad}")?;
            stat.pretty_print(f, pad)?;
            write!(f, "\n")?;
        }
        if let Some(retstat) = &self.retstat {
            write!(f, "{pad}return")?;
            let mut prefix = "";
            for val in &**retstat {
                write!(f, " {prefix}")?;
                val.pretty_print(f, pad)?;
                prefix = ",";
            }
            write!(f, "\n")?;
        }
        Ok(())
    }
}

impl fmt::Display for Block<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.pretty_print(f, &String::new())
    }
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

impl Stat<'_> {
    pub fn pretty_print(&self, f: &mut fmt::Formatter, pad: &String) -> fmt::Result {
        match self {
            Self::Empty => f.write_char(';'),
            Self::Assign { vars, exps } => {
                let mut prefix = "";
                for var in &**vars {
                    write!(f, "{prefix}")?;
                    var.pretty_print(f, pad)?;
                    prefix = ", ";
                }
                f.write_str(" = ")?;
                prefix = "";
                for exp in &**exps {
                    write!(f, "{prefix}")?;
                    exp.pretty_print(f, pad)?;
                    prefix = ", ";
                }
                Ok(())
            }
            Self::FunctionCall(x) => x.pretty_print(f, pad),
            Self::If {
                main: (main_cond, main_block),
                elseifs,
                else_block,
            } => {
                let new_pads = pad.clone() + "    ";
                write!(f, "if ")?;
                main_cond.pretty_print(f, pad)?;
                write!(f, " then\n")?;
                main_block.pretty_print(f, &new_pads)?;
                for (e_cond, e_block) in elseifs {
                    write!(f, "{pad}elseif ")?;
                    e_cond.pretty_print(f, pad)?;
                    write!(f, " then\n")?;
                    e_block.pretty_print(f, &new_pads)?;
                }
                if let Some(e_block) = else_block {
                    write!(f, "{pad}else\n")?;
                    e_block.pretty_print(f, &new_pads)?;
                }
                write!(f, "{pad}end")
            }
            Self::Function { name, body } => {
                write!(f, "function {name}")?;
                body.pretty_print(f, pad)
            }
            x => todo!("{x:?}"),
        }
    }
}

impl fmt::Display for Stat<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.pretty_print(f, &String::new())
    }
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

impl fmt::Display for FuncName<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut prefix = "";
        for comp in &**self.path {
            write!(f, "{prefix}{comp}")?;
            prefix = ".";
        }
        if let Some(method) = &self.method {
            write!(f, ":{method}")?;
        }
        Ok(())
    }
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

impl Var<'_> {
    pub fn pretty_print(&self, f: &mut fmt::Formatter, pad: &String) -> fmt::Result {
        match self {
            Self::Name(x) => f.write_str(x),
            Self::Path { lhs, member } => {
                lhs.pretty_print(f, pad)?;
                write!(f, ".{member}")
            }
            x => todo!("{x:?}"),
        }
    }
}

impl fmt::Display for Var<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.pretty_print(f, &String::new())
    }
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

impl Exp<'_> {
    pub fn pretty_print(&self, f: &mut fmt::Formatter, pad: &String) -> fmt::Result {
        match self {
            Self::Nil => write!(f, "nil"),
            Self::NumeralInt(x) => write!(f, "{x}"),
            Self::NumeralFloat(x) => write!(f, "{x}"),
            Self::LiteralString(x) => write!(f, "{:?}", str::from_utf8(x).unwrap()), // TODO: Write a correct string formatter. This is a cheat.
            Self::PrefixExp(x) => x.pretty_print(f, pad),
            Self::TableConstructor(list) => {
                f.write_str("{")?;
                let mut prefix = "";
                for field in &**list {
                    write!(f, "{prefix} ")?;
                    field.pretty_print(f, pad)?;
                    prefix = ",";
                }
                f.write_str(" }")
            }
            Self::BinExp { lhs, op, rhs } => {
                lhs.pretty_print(f, pad)?;
                write!(f, " {op} ")?;
                rhs.pretty_print(f, pad)
            }
            x => todo!("{x:?}"),
        }
    }
}

impl fmt::Display for Exp<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.pretty_print(f, &String::new())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum PrefixExp<'src> {
    Var(Spanned<Var<'src>>),
    FunctionCall(Spanned<FunctionCall<'src>>),
    Parens(Box<Spanned<Exp<'src>>>),
}

impl PrefixExp<'_> {
    pub fn pretty_print(&self, f: &mut fmt::Formatter, pad: &String) -> fmt::Result {
        match self {
            Self::Var(x) => x.pretty_print(f, pad),
            Self::FunctionCall(x) => x.pretty_print(f, pad),
            x => todo!("{x:?}"),
        }
    }
}

impl fmt::Display for PrefixExp<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.pretty_print(f, &String::new())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct FunctionCall<'src> {
    pub lhs: Box<Spanned<PrefixExp<'src>>>,
    pub method: Option<Spanned<&'src str>>,
    pub args: Spanned<Args<'src>>,
}

impl FunctionCall<'_> {
    pub fn pretty_print(&self, f: &mut fmt::Formatter, pad: &String) -> fmt::Result {
        self.lhs.pretty_print(f, pad)?;
        if let Some(method) = &self.method {
            write!(f, ":{method}")?;
        }
        self.args.pretty_print(f, pad)
    }
}

impl fmt::Display for FunctionCall<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.pretty_print(f, &String::new())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Args<'src> {
    List(List<Exp<'src>>),
    TableConstructor(List<Field<'src>>),
    String(Spanned<Box<[u8]>>),
}

impl Args<'_> {
    pub fn pretty_print(&self, f: &mut fmt::Formatter, pad: &String) -> fmt::Result {
        match self {
            Self::List(list) => {
                write!(f, "(")?;
                let mut prefix = "";
                for x in &**list {
                    write!(f, "{prefix}")?;
                    x.pretty_print(f, pad)?;
                    prefix = ", ";
                }
                write!(f, ")")
            }
            x => todo!("{x:?}"),
        }
    }
}

impl fmt::Display for Args<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.pretty_print(f, &String::new())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct FuncBody<'src> {
    pub params: List<&'src str>,
    pub varargs: Option<Spanned<()>>,
    pub block: Spanned<Block<'src>>,
}

impl FuncBody<'_> {
    pub fn pretty_print(&self, f: &mut fmt::Formatter, pad: &String) -> fmt::Result {
        write!(f, "(")?;
        let mut prefix = "";
        for param in &**self.params {
            write!(f, "{prefix}{param}")?;
            prefix = ", ";
        }
        if self.varargs.is_some() {
            write!(f, "{prefix}...")?;
        }
        write!(f, ")\n")?;
        let new_pad = pad.clone() + "    ";
        self.block.pretty_print(f, &new_pad)?;
        write!(f, "{pad}end")
    }
}

impl fmt::Display for FuncBody<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.pretty_print(f, &String::new())
    }
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

    pub fn pretty_print(&self, f: &mut fmt::Formatter, pad: &String) -> fmt::Result {
        match self {
            Self::Exp { field, val } => {
                f.write_char('[')?;
                field.pretty_print(f, pad)?;
                f.write_str("] = ")?;
                val.pretty_print(f, pad)
            }
            Self::Named { field, val } => {
                write!(f, "{field} = ")?;
                val.pretty_print(f, pad)
            }
            Self::Unnamed { val } => val.pretty_print(f, pad),
        }
    }
}

impl fmt::Display for Field<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.pretty_print(f, &String::new())
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

impl fmt::Display for BinOp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(match self {
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::Mul => "*",
            BinOp::Div => "/",
            BinOp::Idiv => "//",
            BinOp::Pow => "^",
            BinOp::Mod => "%",
            BinOp::Band => "&",
            BinOp::Bxor => "~",
            BinOp::Bor => "|",
            BinOp::Shr => ">>",
            BinOp::Shl => "<<",
            BinOp::Concat => "..",
            BinOp::Lt => "<",
            BinOp::Le => "<=",
            BinOp::Gt => ">",
            BinOp::Ge => ">=",
            BinOp::Eq => "==",
            BinOp::Neq => "~=",
            BinOp::And => "and",
            BinOp::Or => "or",
        })
    }
}

// most names are based on the related metatable function
#[derive(Clone, Debug, PartialEq, Hash, Eq)]
pub enum UnOp {
    Unm,
    Not, // keyword/boolean
    Len,
    Bnot,
}

pub fn parse_string(x: &str) -> Box<[u8]> {
    let mut iter = x.bytes();
    assert_eq!(iter.next(), Some(b'"'));
    let mut result = Vec::new();
    while let Some(x) = iter.next() {
        if x == b'"' {
            break;
        } else if x == b'\\' {
            todo!()
        } else {
            result.push(x);
        }
    }
    result.into_boxed_slice()
}
