use alloc::alloc::AllocError;
use alloc::alloc::Allocator;
use bytemuck::Zeroable;
use core::alloc::Layout;
use core::cell::{Cell, UnsafeCell};
use core::hash::{Hash, Hasher};
use core::marker::PhantomData;
use core::mem::MaybeUninit;
use core::ops::Deref;
use core::ptr::NonNull;

use crate::engine::table::Table;
use crate::mir::FunctionDef;
use crate::sync::RwLock;

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

const TYPE_MASK: u64 = 0xF000_0000_0000_0000;

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
    NullTy = 0,
    Table = 1,
    Closure = 2,
    ManagedString = 3,
    Coroutine,
    ManagedUserdata,
    UpvarSpan,
    Dangling = 7,

    FinalizedManagedUserdata = 0xA,
    FinalizedCoroutine = 0xB,
    FinalizedTable = 0xE,
    Dead = 0xF,
}

#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct Value<'ctx>(ValueInner, Brand<'ctx>);

unsafe impl<'ctx> Zeroable for Value<'ctx> {}

const STRING_LEN_MASK: usize = 0x3FFF_FFFF_FFFF_FFFFu64 as usize;

impl<'ctx> Value<'ctx> {
    const fn new_unchecked(x: ValueInner) -> Self {
        Self(x, Brand::new_unchecked())
    }

    const fn from_ptr<T: ArenaTy<'ctx>>(v: ArenaPtr<'ctx, T>) -> Self {
        let sz = const {
            let sz = core::mem::size_of::<T>();

            if sz > u32::MAX as usize {
                panic!("OVerlarge values not allowed for managed values")
            }
            sz as u32
        };
        Self::new_unchecked(ValueInner {
            wide_ptr: ValuePtr {
                ptr: core::ptr::without_provenance_mut(v.0),
                __pad: bytemuck::zeroed(),
                meta: TYPE_MANAGED | ((T::TY as u64) << 56) | sz as u64,
            },
        })
    }

    pub const fn dead() -> Self {
        Self::new_unchecked(ValueInner {
            int: ValueInt { int: !0, meta: !0 },
        })
    }

    pub const fn nil() -> Self {
        Self::new_unchecked(ValueInner {
            wide_ptr: ValuePtr {
                ptr: core::ptr::null_mut(),
                __pad: bytemuck::zeroed(),
                meta: 0,
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
                meta: TYPE_FLOAT,
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

    pub const fn string_literal(x: &'ctx [u8]) -> Self {
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
            0 if unsafe { self.0.int.int == 0 } => Type::Managed(ManagedType::NullTy),
            0..8 => Type::UnmanagedString,
            8 => Type::Int,
            9 => Type::Float,
            10 => Type::Boolean,
            15 => Type::Managed(ManagedType::from_int((meta >> 56) as usize & 0xF)),
            _ => panic!("Invalid type encoded in raw representation"),
        }
    }

    pub fn unpack(&self) -> UnpackedValue<'ctx> {
        match self.type_of() {
            Type::Managed(managed_type) => {
                // SAFETY:
                // This field is live, and known to contain a `usize` value (empty provenance), so this is DB
                let aptr = unsafe { self.0.wide_ptr.ptr.addr() };
                let mval = match managed_type {
                    ManagedType::NullTy => ManagedValue::Nil,
                    ManagedType::Table => {
                        ManagedValue::Table(unsafe { ArenaPtr::new_unchecked(aptr) })
                    }
                    ManagedType::Closure => {
                        ManagedValue::Closure(unsafe { ArenaPtr::new_unchecked(aptr) })
                    }
                    ManagedType::ManagedString => {
                        ManagedValue::String(unsafe { ArenaPtr::new_unchecked(aptr) })
                    }
                    ManagedType::Coroutine => todo!(),
                    ManagedType::ManagedUserdata => todo!(),
                    ManagedType::UpvarSpan => unreachable!(),
                    ManagedType::Dangling => ManagedValue::Nil,
                    ManagedType::FinalizedManagedUserdata => todo!(),
                    ManagedType::FinalizedCoroutine => todo!(),
                    ManagedType::FinalizedTable => todo!(),
                    ManagedType::Dead => unreachable!(),
                };

                UnpackedValue::Managed(mval)
            }
            Type::Boolean => todo!(),
            Type::Int => todo!(),
            Type::Float => todo!(),
            Type::UnmanagedString => UnpackedValue::String(unsafe {
                core::slice::from_raw_parts(
                    self.0.wide_ptr.ptr.cast(),
                    self.0.wide_ptr.meta as usize & STRING_LEN_MASK,
                )
            }),
        }
    }

    pub fn is_nil(&self) -> bool {
        let ptr = unsafe { self.0.wide_ptr };

        ptr.ptr.is_null() && (ptr.meta & TYPE_MASK == TYPE_MANAGED || ptr.meta & TYPE_MASK == 0)
    }

    pub fn bool_test(&self) -> bool {
        let valint = unsafe { self.0.int };

        !(valint.int == 0
            && (valint.meta & TYPE_MASK == TYPE_BOOL
                || valint.meta & TYPE_MASK == TYPE_MANAGED
                || valint.meta & TYPE_MASK == 0))
    }

    pub const fn as_int(&self) -> Option<i64> {
        match self.type_of() {
            Type::Int => Some(unsafe { self.0.int.int as i64 }),
            Type::Float => {
                let x = unsafe { f64::from_bits(self.0.int.int) };
                let y = x as i64;

                if x == (y as f64) { Some(y) } else { None }
            }
            _ => None,
        }
    }

    pub fn normalize(self) -> Self {
        if self.is_nil() {
            Self::nil()
        } else {
            match self.type_of() {
                Type::Managed(ManagedType::NullTy | ManagedType::Dangling | ManagedType::Dead) => {
                    Self::nil()
                }
                _ => self, // TODO: Also Normalize NaNs
            }
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum UnpackedValue<'ctx> {
    Int(i64),
    Bool(bool),
    Float(f64),
    String(&'ctx [u8]),
    Managed(ManagedValue<'ctx>),
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum ManagedValue<'ctx> {
    Nil,
    Table(ArenaPtr<'ctx, Table<'ctx>>),
    Closure(ArenaPtr<'ctx, LuaFunction<'ctx>>),
    String(ArenaPtr<'ctx, Box<'ctx, [u8]>>),
    // More For later
}

pub struct ArenaPtr<'ctx, T>(usize, Brand<'ctx>, PhantomData<*mut T>);

impl<'ctx, T> ArenaPtr<'ctx, T> {
    pub const fn null() -> Self {
        Self(0, Brand::new_unchecked(), PhantomData)
    }
    pub const unsafe fn new_unchecked(offset: usize) -> Self {
        unsafe {
            core::hint::assert_unchecked(offset & (align_of::<ValueBlock>() - 1) == 0);
        }

        Self(offset, Brand::new_unchecked(), PhantomData)
    }
}

impl<'ctx, T> Copy for ArenaPtr<'ctx, T> {}
impl<'ctx, T> Clone for ArenaPtr<'ctx, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'ctx, T> PartialEq for ArenaPtr<'ctx, T> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<'ctx, T> Eq for ArenaPtr<'ctx, T> {}

impl<'ctx, T> core::fmt::Debug for ArenaPtr<'ctx, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        core::ptr::without_provenance_mut::<T>(self.0).fmt(f)
    }
}

impl<'ctx, T> core::fmt::Pointer for ArenaPtr<'ctx, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        core::ptr::without_provenance_mut::<T>(self.0).fmt(f)
    }
}

#[repr(C, align(32))]
struct ValueBlock(UnsafeCell<MaybeUninit<[u8; 32]>>);

unsafe impl Send for ValueBlock {}
unsafe impl Sync for ValueBlock {}

struct ArenaInner {
    // Starts with the allocation Metadata, The rest is the Memory region
    x: NonNull<[ValueBlock]>,
}

unsafe impl Send for ArenaInner {}
unsafe impl Sync for ArenaInner {}

const USIZE_PER_BLOCK: usize = 32 / core::mem::size_of::<usize>();

impl ArenaInner {
    fn create_arena(len: usize) -> Self {
        assert!(len > 1);
        let layout = Layout::array::<ValueBlock>(len).unwrap();
        let a = unsafe { alloc::alloc::alloc_zeroed(layout) };

        if let Some(a) = NonNull::new(a) {
            Self {
                x: NonNull::slice_from_raw_parts(a.cast::<ValueBlock>(), len),
            }
        } else {
            alloc::alloc::handle_alloc_error(layout)
        }
    }

    pub fn get_or_init_header(&mut self) -> &mut AllocMetadata {
        let mut head_size = unsafe { core::ptr::read(self.x.cast::<usize>().as_ptr()) };
        if head_size == 0 {
            let block_count = self.x.len();

            head_size = (block_count + usize::BITS as usize).div_ceil(256);

            unsafe {
                core::ptr::write(self.x.cast::<[usize; 2]>().as_ptr(), [head_size, head_size]);
            }
        }

        let len = (head_size - 1) * USIZE_PER_BLOCK;

        unsafe {
            &mut *(core::ptr::slice_from_raw_parts_mut(self.x.cast::<usize>().as_ptr(), len)
                as *mut AllocMetadata)
        }
    }
}

impl Drop for ArenaInner {
    fn drop(&mut self) {
        unsafe {
            drop(alloc::boxed::Box::from_raw(self.x.as_ptr()));
        }
    }
}

struct AllocMetadata {
    head_size: usize,
    next_block: usize,
    first_block: [usize; USIZE_PER_BLOCK - 2],
    rest_blocks: [usize],
}

pub struct Arena<'ctx>(RwLock<ArenaInner>, Brand<'ctx>);

impl<'ctx> Arena<'ctx> {
    pub fn resolve_ptr<T>(&self, ptr: ArenaPtr<'ctx, T>) -> *mut T {
        unsafe { self.0.read().x.as_ptr().cast::<T>().byte_add(ptr.0) }
    }

    pub fn try_allocate_arena<T>(&self) -> Result<ArenaPtr<'ctx, T>, AllocError> {
        let raw = self.allocate(Layout::new::<T>())?.cast::<T>();

        let ptr = unsafe {
            self.0
                .read()
                .x
                .as_ptr()
                .cast::<T>()
                .byte_offset_from_unsigned(raw.as_ptr())
        };
        debug_assert!((ptr & 31) == 0);
        unsafe {
            core::hint::assert_unchecked((ptr & 31) == 0);
        }

        let ptr = ArenaPtr(ptr, Brand::new_unchecked(), PhantomData);

        Ok(ptr)
    }

    pub fn try_allocate_managed_value<T: ArenaTy<'ctx>>(
        &self,
        val: T,
    ) -> Result<Value<'ctx>, AllocError> {
        let ptr = self.try_allocate_arena::<T>()?;
        unsafe {
            self.resolve_ptr(ptr).write(val);
        }

        Ok(Value::from_ptr::<T>(ptr))
    }

    pub fn allocate_managed_value<T: ArenaTy<'ctx>>(&self, val: T) -> Value<'ctx> {
        self.try_allocate_managed_value(val)
            .unwrap_or_else(|_| alloc::alloc::handle_alloc_error(Layout::new::<T>()))
    }
}

unsafe impl<'ctx> Allocator for Arena<'ctx> {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, alloc::alloc::AllocError> {
        if layout.align() > align_of::<ValueBlock>() {
            return Err(AllocError);
        }

        let size = layout.size().div_ceil(32);

        let mut lock = self.0.write();
        let base = lock.x;
        let meta = lock.get_or_init_header();

        if base.len() < (size + meta.next_block) {
            return Err(AllocError);
        }

        let blocks = unsafe { base.cast::<ValueBlock>().add(meta.next_block) };
        meta.next_block += size;

        Ok(NonNull::slice_from_raw_parts(
            blocks.cast::<u8>(),
            size * 32,
        ))
    }

    unsafe fn deallocate(&self, _: core::ptr::NonNull<u8>, _: Layout) {
        // Bump allocator for now
    }
}

pub unsafe trait Gcable<'ctx> {
    unsafe fn mark_reachables(
        &self,
        reachable_span: &mut dyn FnMut(*const [u8]),
        reachable_value: &mut dyn FnMut(Value<'ctx>),
    );
}

pub unsafe trait ArenaTy<'ctx> {
    const TY: ManagedType;
}

pub mod table;

pub struct LuaEngine<'ctx> {
    arena: Arena<'ctx>,
    global_env: Cell<Value<'ctx>>,
}

unsafe impl<'ctx> Send for LuaEngine<'ctx> {}
unsafe impl<'ctx> Sync for LuaEngine<'ctx> {}

impl<'ctx> Deref for LuaEngine<'ctx> {
    type Target = Arena<'ctx>;

    fn deref(&self) -> &Arena<'ctx> {
        &self.arena
    }
}

impl LuaEngine<'static> {
    pub fn with_userdata<T, R>(
        arena_size: usize,
        udata: &T,
        with_fn: impl for<'ctx> FnOnce(&'ctx LuaEngine<'ctx>, &'ctx T) -> R,
        populate_env: impl for<'ctx> FnOnce(&mut Table<'ctx>, &'ctx LuaEngine<'ctx>),
    ) -> R {
        let arena_size = arena_size / size_of::<ValueBlock>();

        let arena = Arena(
            RwLock::new(ArenaInner::create_arena(arena_size)),
            Brand::new_unchecked(),
        );
        let engine = LuaEngine {
            arena,
            global_env: Cell::new(Value::nil()),
        };

        let mut table = Table::new(&engine);

        populate_env(&mut table, &engine);
        let table = engine.allocate_managed_value(table);
        match table.unpack() {
            UnpackedValue::Managed(ManagedValue::Table(t)) => unsafe {
                (*engine.resolve_ptr(t)).insert(&engine, Value::string_literal(b"_G"), table);
            },
            _ => {}
        }
        engine.global_env.set(table);

        with_fn(&engine, udata)
    }

    pub fn with<R>(
        arena_size: usize,
        with_fn: impl for<'ctx> FnOnce(&'ctx LuaEngine<'ctx>) -> R,
        populate_env: impl for<'ctx> FnOnce(&mut Table<'ctx>, &'ctx LuaEngine<'ctx>),
    ) -> R {
        Self::with_userdata(arena_size, &(), move |a, _| with_fn(a), populate_env)
    }
}

impl<'ctx> LuaEngine<'ctx> {
    pub fn alloc(&self) -> &Arena<'ctx> {
        &self.arena
    }

    pub fn create_closure(
        &'ctx self,
        def: FunctionDef,
        captures: CaptureSpan<'ctx>,
    ) -> Value<'ctx> {
        self.allocate_managed_value(LuaFunction::Lua(Closure { def, captures }))
    }

    //
    fn as_string(&self, val: Value<'ctx>) -> Option<&[u8]> {
        match val.unpack() {
            UnpackedValue::String(s) => Some(s),
            UnpackedValue::Managed(ManagedValue::String(s)) => {
                let ptr = unsafe { &*self.resolve_ptr(s) };
                Some(ptr)
            }
            _ => None,
        }
    }

    pub fn raw_hash<H: Hasher>(&self, val: Value<'ctx>, state: &mut H) {
        use core::hash::Hash;
        let ty = val.type_of();

        core::mem::discriminant(&ty).hash(state);

        match ty {
            Type::Int | Type::Boolean | Type::Float => unsafe {
                val.0.int.int.hash(state);
            },
            Type::UnmanagedString => {
                let len = unsafe { val.0.wide_ptr.meta as usize & STRING_LEN_MASK };

                hash_string(
                    unsafe { core::slice::from_raw_parts(val.0.wide_ptr.ptr.cast::<u8>(), len) },
                    state,
                );
            }
            Type::Managed(ManagedType::ManagedString) => {
                let ptr =
                    unsafe { ArenaPtr::<Box<[u8]>>::new_unchecked(val.0.wide_ptr.ptr.addr()) };

                let sl = unsafe { &*self.resolve_ptr(ptr) };

                hash_string(sl, state);
            }
            Type::Managed(_) => core::ptr::hash(unsafe { val.0.wide_ptr.ptr }, state),
        }
    }

    pub fn raw_eq_lookup(&self, a: Value<'ctx>, b: Value<'ctx>) -> bool {
        let ty1 = a.type_of();
        let ty2 = b.type_of();

        match (ty1, ty2) {
            (Type::Int | Type::Float, Type::Int | Type::Float) => {
                let val1 = unsafe { a.0.int.int };
                let val2 = unsafe { a.0.int.int };

                if val1 == val2 {
                    true
                } else if val1 & 0x8000_0000_0000_0000 != val2 & 0x8000_0000_0000_0000 {
                    // This is a check specifically for -0.0 == 0 (int)/ 0 == -0.0
                    false
                } else {
                    match (a.as_int(), b.as_int()) {
                        (Some(a), Some(b)) => a == b,
                        _ => false,
                    }
                }
            }
            (
                Type::UnmanagedString | Type::Managed(ManagedType::ManagedString),
                Type::UnmanagedString | Type::Managed(ManagedType::ManagedString),
            ) => self.as_string(a) == self.as_string(b),

            (at @ Type::Managed(_), bt @ Type::Managed(_)) if at == bt => unsafe {
                a.0.wide_ptr.ptr == b.0.wide_ptr.ptr
            },
            (at, bt) if at == bt => unsafe { a.0.int.int == b.0.int.int },
            _ => false,
        }
    }
}

fn hash_string<H: Hasher>(x: &[u8], state: &mut H) {
    let len_diff = 8 - x.len() & 7;
    state.write(x);
    state.write(&[!x.last().copied().unwrap_or(0); 8][..len_diff])
}

pub struct Closure<'ctx> {
    def: FunctionDef,
    captures: CaptureSpan<'ctx>,
}

pub enum LuaFunction<'ctx> {
    Lua(Closure<'ctx>),
    Rust(Box<'ctx, dyn LuaCallable<'ctx> + 'ctx>),
}

unsafe impl<'ctx> ArenaTy<'ctx> for LuaFunction<'ctx> {
    const TY: ManagedType = ManagedType::Closure;
}

pub enum CaptureSpan<'ctx> {
    Direct(Vec<'ctx, Value<'ctx>>),
    Indirect(Vec<'ctx, CaptureSpan<'ctx>>),
}

pub type Vec<'ctx, T> = alloc::vec::Vec<T, &'ctx Arena<'ctx>>;
pub type Box<'ctx, T> = alloc::boxed::Box<T, &'ctx Arena<'ctx>>;

pub trait LuaCallable<'ctx> {
    fn call(
        &mut self,
        engine: &'ctx LuaEngine<'ctx>,
        params: &[Value<'ctx>],
    ) -> Result<Vec<'ctx, Value<'ctx>>, LuaError<'ctx>>;
}

pub struct LuaError<'ctx>(Brand<'ctx>); // For now
