use core::alloc::Layout;
use core::cell::UnsafeCell;
use core::hash::{Hash, Hasher};
use core::marker::PhantomData;
use core::mem::MaybeUninit;

use crate::sync::RwLock;

use alloc::boxed::Box;

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct Brand<'ctx>(PhantomData<&'ctx mut &'ctx mut ()>);

impl<'ctx> Brand<'ctx> {
    const fn new_unchecked() -> Self {
        Self(PhantomData)
    }
}

#[derive(Copy, Clone)]
#[repr(C)]
struct ValuePtr {
    ptr: *mut (),
    __pad: [usize; (size_of::<u64>() / size_of::<usize>()) - 1],
    meta: u64,
}

unsafe impl Send for ValuePtr {}
unsafe impl Sync for ValuePtr {}

#[derive(Copy, Clone, Hash, PartialEq, Eq)]
#[repr(C)]
struct ValueInt {
    int: u64,
    meta: u64,
}

#[derive(Copy, Clone)]
#[repr(C, align(16))]
union ValueInner {
    wide_ptr: ValuePtr,
    int: ValueInt,
}

impl Hash for ValueInner {
    fn hash<H: Hasher>(&self, state: &mut H) {
        unsafe { self.int.hash(state) }
    }
}

impl PartialEq for ValueInner {
    fn eq(&self, other: &Self) -> bool {
        unsafe { self.int == other.int }
    }
}

impl Eq for ValueInner {}

const TYPE_INT: u64 = 0x8000_0000_0000_0000;
const TYPE_FLOAT: u64 = 0x9000_0000_0000_0000;
const TYPE_BOOL: u64 = 0xA000_0000_0000_0000;
const TYPE_UNMANAGED_UDATA: u64 = 0xB000_0000_0000_0000;

const TYPE_MANAGED: u64 = 0xF000_0000_0000_0000;

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum Type {
    Managed(ManagedType),
    Boolean,
    Int,
    Float,
    UnmanagedString,
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
    ManagedString = 3,
    Coroutine,
    ManagedUserdata,
    Dangling = 7,

    FinalizedManagedUserdata = 0xA,
    FinalizedCoroutine = 0xB,
    FinalizedTable = 0xE,
    Dead = 0xF,
}

#[repr(transparent)]
#[derive(Copy, Clone, Hash, PartialEq, Eq)]
pub struct Value<'ctx>(ValueInner, Brand<'ctx>);

impl<'ctx> Value<'ctx> {
    const fn new_unchecked(x: ValueInner) -> Self {
        Self(x, Brand::new_unchecked())
    }
    pub const fn nil() -> Self {
        Self::new_unchecked(ValueInner {
            wide_ptr: ValuePtr {
                ptr: core::ptr::null_mut(),
                __pad: bytemuck::zeroed(),
                meta: TYPE_MANAGED | (ManagedType::Dangling as u64) << 56,
            },
        })
    }

    pub const fn new_int(x: i64) -> Self {
        Self::new_unchecked(ValueInner {
            int: ValueInt {
                int: x as u64,
                meta: TYPE_INT,
            },
        })
    }

    pub const fn new_float(x: f64) -> Self {
        Self::new_unchecked(ValueInner {
            int: ValueInt {
                int: x as u64,
                meta: TYPE_INT,
            },
        })
    }

    pub const fn new_bool(x: bool) -> Self {
        Self::new_unchecked(ValueInner {
            int: ValueInt {
                int: x as u64,
                meta: TYPE_BOOL,
            },
        })
    }

    pub const fn string_literal(x: &'static [u8]) -> Self {
        #[cfg(target_pointer_width = "64")]
        assert!(x.len() < 0x4000_0000_0000_0000);
        let ptr = x.as_ptr();
        let len = x.len();

        Self::new_unchecked(ValueInner {
            wide_ptr: ValuePtr {
                ptr: ptr.cast_mut().cast(),
                __pad: bytemuck::zeroed(),
                meta: len as u64,
            },
        })
    }

    pub const fn managed_size(&self) -> Option<usize> {
        let meta = unsafe { self.0.wide_ptr.meta };
        if (meta & 0xF000_0000_0000_0000) == 0xF000_0000_0000_0000 {
            Some(meta as u32 as usize)
        } else if meta < 0x8000_0000_0000_0000 {
            Some((meta & 0x3FFF_FFFF_FFFF_FFFF) as usize)
        } else {
            None
        }
    }

    pub const fn type_of(&self) -> Type {
        let meta = unsafe { self.0.wide_ptr.meta };

        match meta >> 60 {
            0..8 => Type::UnmanagedString,
            8 => Type::Int,
            9 => Type::Float,
            10 => Type::Boolean,
            15 => Type::Managed(ManagedType::from_int((meta >> 56) as usize & 0xF)),
            _ => panic!("Invalid type encoded in raw representation"),
        }
    }

    pub const fn unpack(&self) -> UnpackedValue<'ctx> {
        todo!()
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum UnpackedValue<'ctx> {
    Int(i32),
    Bool(bool),
    Float(f64),
    String(&'ctx [u8]),
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
struct ValueBlock(UnsafeCell<MaybeUninit<[u8; 32]>>);

unsafe impl Send for ValueBlock {}
unsafe impl Sync for ValueBlock {}

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

impl<'ctx> Arena<'ctx> {}
