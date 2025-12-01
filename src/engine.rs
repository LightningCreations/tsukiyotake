use core::alloc::Layout;
use core::marker::PhantomData;
use core::mem::MaybeUninit;

use crate::sync::RwLock;

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct Brand<'ctx>(PhantomData<&'ctx mut &'ctx mut ()>);

#[repr(align(8))]
#[derive(Copy, Clone, Hash, PartialEq, Eq)]
pub struct Value<'ctx>(u64, Brand<'ctx>);

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum Type {
    Managed(ManagedType),
    Boolean,
    Int,
    Float,
}

macro_rules! TryFromInt {
    derive() ($(#[$meta:meta])* $vis:vis enum $name:ident { $($var:ident $(= $expr:expr)?),* $(,)?}) => {
        impl $name {
            #[allow(non_upper_case_globals)]
            const fn from_int(x: usize) -> Self {
                $(pub const $var: usize = $name::$var as usize;)*

                match x {
                    $($var => Self::$var,)*
                    _ => panic!(core::concat!("Invalid value for ", core::stringify!($name)))
                }
            }
        }
    };
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq, TryFromInt)]
pub enum ManagedType {
    Table = 1,
    StaticString,
    ManagedString,
    Coroutine,
    ManagedUserdata,
    UnmanagedUserdata,
    Dangling,

    FinalizedUnmanagedUserdata = 0xA,
    FinalizedManagedUserdata = 0xB,
    FinalizedCoroutine = 0xC,
    FinalizedTable = 0xE,
    Dead = 0xF,
}

impl ManagedType {
    const MAX_VALUE: u64 = ManagedType::UnmanagedUserdata as u64;
}

const NAN: u64 = 0x7FF0_0000_0000_0001;

const NIL: u64 = 0x7FF7_0000_0000_0000;

impl<'ctx> Value<'ctx> {
    pub const fn new_int(x: i32) -> Value<'ctx> {
        Value(0x7FFF_0000_0000_0000 | x as u32 as u64, Brand(PhantomData))
    }

    pub const fn new_float(x: f64) -> Value<'ctx> {
        if x.is_nan() {
            Value(NAN, Brand(PhantomData))
        } else {
            Value(x.to_bits(), Brand(PhantomData))
        }
    }

    pub const fn nil() -> Value<'ctx> {
        Value(NIL, Brand(PhantomData))
    }

    pub const fn new_bool(x: bool) -> Value<'ctx> {
        Value(0x7FFE_0000_0000_0000 | x as u64, Brand(PhantomData))
    }

    pub const fn type_of(&self) -> Type {
        match self.0 >> 48 {
            0x7FFF => Type::Int,
            0x7FFE => Type::Boolean,
            0xFFF1..=0xFFFF | 0x7FF1..=0x7FFD => {
                let x = (self.0 >> 48) & 0xF;

                Type::Managed(ManagedType::from_int(x as usize))
            }
            _ => Type::Float,
        }
    }

    pub const fn is_gc_marked(&self) -> bool {
        self.0 >> 63 == 1
    }

    pub const fn raw(&self) -> u64 {
        self.0
    }

    pub const fn unpack(&self) -> UnpackedValue<'ctx> {
        match self.type_of() {
            Type::Int => UnpackedValue::Int(self.0 as u32 as i32),
            Type::Boolean => UnpackedValue::Bool(self.0 & 0x0000_0000_0000 != 0),
            Type::Float => UnpackedValue::Float(f64::from_bits(self.0)),
            Type::Managed(m) => UnpackedValue::Managed(ArenaPtr(
                ((self.0 & 0xFFFF_FFFF_FFFF) as usize) << 5 | (m as usize),
                self.1,
            )),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum UnpackedValue<'ctx> {
    Int(i32),
    Bool(bool),
    Float(f64),
    Managed(ArenaPtr<'ctx>),
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct ArenaPtr<'ctx>(usize, Brand<'ctx>);

impl<'ctx> ArenaPtr<'ctx> {
    pub const fn type_of(&self) -> ManagedType {
        ManagedType::from_int(self.0 & 0xF)
    }
}

#[repr(C, align(32))]
struct ValueBlock(MaybeUninit<[u8; 32]>);

struct ArenaInner<'ctx> {
    // Starts with the allocation Metadata, The rest is the Memory region
    x: Box<[ValueBlock]>,
    ctx: Brand<'ctx>,
}

struct AllocMetadata {
    head_size: usize,
    first_block: [usize; (32 / core::mem::size_of::<usize>()) - 1],
    rest_blocks: [usize],
}

pub struct Arena<'ctx>(RwLock<ArenaInner<'ctx>>);
