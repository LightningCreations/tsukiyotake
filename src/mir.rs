use alloc::collections::BTreeMap;
use core::{fmt, num::NonZeroU32};

use crate::ast::Spanned;
use alloc::{boxed::Box, string::String, vec::Vec};

use crate::ast::{BinOp, UnOp};

mod from_hir;
pub use from_hir::*;

#[derive(Copy, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct BasicBlockId(NonZeroU32);

impl BasicBlockId {
    pub(crate) const fn new_unchecked(x: NonZeroU32) -> Self {
        Self(x)
    }

    pub const fn next(self) -> Self {
        Self(self.0.checked_add(1).unwrap())
    }

    pub const UNUSED: BasicBlockId = Self(nz!(0xFFFF_FFFF));
}

impl fmt::Display for BasicBlockId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_fmt(format_args!("%{}", self.0))
    }
}

impl fmt::Debug for BasicBlockId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_fmt(format_args!("%{}", self.0))
    }
}

#[macro_export]
macro_rules! bb {
    ($x:expr) => {
        BasicBlockId::new_unchecked(NonZeroU32::new($x).unwrap())
    };
}

#[derive(Copy, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct SsaVarId(NonZeroU32);

impl SsaVarId {
    pub(crate) const fn new_unchecked(x: NonZeroU32) -> Self {
        Self(x)
    }

    pub const fn next(self) -> Self {
        Self(self.0.checked_add(1).unwrap())
    }

    pub const _ENV: Self = Self(nz!(1));
}

impl fmt::Display for SsaVarId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_fmt(format_args!("${}", self.0))
    }
}

impl fmt::Debug for SsaVarId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_fmt(format_args!("${}", self.0))
    }
}

#[macro_export]
macro_rules! ssa_var {
    ($x:expr) => {
        SsaVarId::new_unchecked(NonZeroU32::new($x).unwrap())
    };
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum Terminator {
    DeferError(Spanned<String>),
    RtError(Spanned<String>, Multival),
    Branch(Expr, JumpTarget, JumpTarget),
    Jump(JumpTarget),
    Tailcall(Spanned<FunctionCall>),
    Return(Multival),
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct JumpTarget {
    pub targ_bb: u32,
    pub remaps: Vec<(SsaVarId, SsaVarId)>,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum Multival {
    Empty,
    /// A fixed list of [`Expr`]
    FixedList(Vec<Expr>),
    /// Gets a multival variable (usually initialized by a function call, but also ... will count). Single val variables use `Expr::Var`
    /// Second field is statically known count or [`None`] if unknown
    Var {
        var: SsaVarId,
        count: Option<u32>,
    },
    /// Concatenates several arbitrarily size [`Multival`].
    Concat(Vec<Multival>),
    /// Clamps an existing multival to have a size between min and max.
    /// if the size of the list is less than `min`, pads with `nil`. If the size is greater than `max` (if not [`None`]), truncates at `max` values
    ClampSize {
        base: Box<Multival>,
        min: u32,
        max: Option<u32>,
    },
    /// Slices a [`Multival`] to start from `start` and end at `end`.
    /// Both must be statically inbounds (use [`Multival::ClampSize`] to ensure this if necessary)
    Subslice {
        base: Box<Multival>,
        start: u32,
        end: Option<u32>,
    },
}

impl Multival {
    pub fn min_size(&self) -> u32 {
        match self {
            Multival::Empty => 0,
            Multival::FixedList(exprs) => exprs.len().try_into().unwrap(),
            Multival::Var { var, count } => count.unwrap_or(0),
            Multival::Concat(multivals) => multivals.iter().map(Multival::min_size).sum(),
            Multival::ClampSize { base, min, .. } => base.min_size().max(*min),
            Multival::Subslice {
                base,
                start: n,
                end: None,
            } => base.min_size() - *n,
            Multival::Subslice {
                base,
                start,
                end: Some(end),
            } => end - start,
        }
    }

    pub fn max_size(&self) -> Option<u32> {
        match self {
            Multival::Empty => Some(0),
            Multival::FixedList(exprs) => Some(exprs.len().try_into().unwrap()),
            Multival::Var { var, count } => *count,
            Multival::Concat(multivals) => multivals
                .iter()
                .map(Multival::max_size)
                .fold(Some(0), |f, g| f.zip(g).map(|(a, b)| a + b)),
            Multival::ClampSize {
                base,
                min,
                max: None,
            } => base.max_size(),
            Multival::ClampSize {
                base,
                min,
                max: Some(val),
            } => Some(base.max_size().map(|v| v.min(*val)).unwrap_or(*val)),
            Multival::Subslice {
                base,
                start,
                end: None,
            } => base.max_size().map(|v| v - *start),
            Multival::Subslice {
                base,
                start,
                end: Some(end),
            } => Some(end - start),
        }
    }

    pub fn exact_size(&self) -> Option<u32> {
        self.max_size().filter(|v| *v == self.min_size())
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum Expr {
    Nil,
    Var(SsaVarId),
    ReadUpvar(SsaVarId),
    Extract(Box<Multival>, u32),
    Table(TableConstructor),
    String(Vec<u8>),
    Closure(ClosureDef),
    Boolean(bool),
    Integer(i64),
    Float(u64),
    Index(Spanned<IndexExpr>),
    UnaryOp(Spanned<UnaryExpr>),
    BinaryOp(Spanned<BinaryExpr>),
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct UnaryExpr {
    pub op: UnOp,
    pub expr: Box<Expr>,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct BinaryExpr {
    pub op: BinOp,
    pub left: Box<Expr>,
    pub right: Box<Expr>,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct IndexExpr {
    pub base: Box<Expr>,
    pub index: Index,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum Index {
    Expr(Box<Expr>),
    Name(String),
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum Statement {
    Multideclare(Vec<SsaVarId>, Multival),
    Call(Option<SsaVarId>, FunctionCall),
    AllocUpvar(SsaVarId, Expr),
    WriteUpvar(SsaVarId, Expr),
    /// Evaluates an Expression including its potential side effects, and discards the result.
    /// Note that this is equivalent to assigning an empty variable list to the [`Multival`] clamped to `0..0`.
    Discard(Expr),
    /// Hints that the SSA Var is not subsequently used by the function. The behaviour depends on the ssa var kind:
    /// * If the var is a single value ssa var, Hints that the heap value, if any, is not subequently used by the function (and can be GCd if no other uses exist)
    /// * If the var is a multival ssa var, Hints that all of the corresponding heap values are no longer used (and can be GCd if no other uses exist)
    /// * If the var is an upvar key, hints that the upvar is not subsequently accessed by the current execution of the function
    ///
    /// It is undefined behaviour to use a GCable value in an SSA Var or to access an upvar after using this operation.
    MarkUnused(SsaVarId),
    /// Hints that the SSA Var is not subequently used by the function.
    /// [`Statement::MarkDead`] has the same behaviour as [`Statement::MarkUnused`], except when applied to an upvar key.
    /// When applied to an upvar key, it hints that the upvar is not subequently accessed by the current **or any future** executions of the function (thus, the upvar slot can be freed).
    ///
    /// Accessing the same upvar from the same closure (whether or not during the same execution) following this statement, is undefined behaviour.
    MarkDead(SsaVarId),
    WriteIndex {
        table: Expr,
        index: Index,
        value: Expr,
    },
    /// Invokes (with a discarded result), close metamethods for each provided variable.
    /// If more than one variable is provided, they are closed in the reverse order they are listed.
    Close(Vec<SsaVarId>),
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct FunctionCall {
    pub base: Expr,
    pub params: Multival,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct TableConstructor {
    pub hash_part: Vec<(Index, Expr)>,
    pub array_part: Box<Multival>,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct ClosureDef {
    pub captures: Vec<SsaVarId>,
    pub function: FunctionDef,
}

/// Assignment of SSA Var numbers:
/// * _ENV is intialially assigned to `_1` always.
/// * Parameters are assigned to `_2` through `_N` (where `N-1` is the number of fixed arity parameters`),
/// * `...` is assigned to `_{N+1}` if present (top level def is always variadic)
/// * Locals can use `each ssa var after that
/// * The upvar keys are assigned in reverse order from the highest numebered ssa vars.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct FunctionDef {
    pub debug_info: FunctionDebugInfo,
    pub num_upvars: u32,
    pub num_params: u32,
    pub variadic: bool,
    pub blocks: Vec<BasicBlock>,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct FunctionDebugInfo {
    pub var_names_map: BTreeMap<SsaVarId, Spanned<String>>,
    pub function_canon_name: Spanned<String>,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct BasicBlock {
    pub id: BasicBlockId,
    pub stats: Vec<Statement>,
    pub term: Terminator,
}
