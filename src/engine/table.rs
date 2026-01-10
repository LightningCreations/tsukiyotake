use core::hash::Hash;
use core::hash::{BuildHasher, Hasher};
use hashbrown::HashTable;
use lccc_siphash::build::RandomState;

use crate::engine::{ArenaPtr, ArenaTy, LuaEngine, ManagedType, Type, Value};

use alloc::alloc::{Allocator, Layout};

use core::ptr::NonNull;

use super::{Box, Vec};

pub struct Table<'ctx> {
    slen: usize,
    htb: HashTable<usize, &'ctx super::Arena<'ctx>>,
    // Before `raw` are array elements
    // After are alternating keys and values
    raw: NonNull<Value<'ctx>>,
    array_cap: usize,
    map_cap: usize,
    next_map_slot: usize,
    metatable: ArenaPtr<'ctx, Table<'ctx>>,
    gc_behaviour: TableGcDisposition,
    hasher: RandomState<2, 4>,
    arena: &'ctx super::Arena<'ctx>,
}

unsafe impl<'ctx> ArenaTy<'ctx> for Table<'ctx> {
    const TY: super::ManagedType = ManagedType::Table;
}

impl<'ctx> Table<'ctx> {
    fn raw_grow(&mut self, new_map_cap: usize, new_array_cap: usize) {
        let total_cap = 2 * new_map_cap + new_array_cap;
        let layout = Layout::array::<Value<'ctx>>(total_cap).unwrap();
        assert!(layout.size() > 0);
        let alloc = self
            .arena
            .allocate_zeroed(layout)
            .unwrap_or_else(|_| alloc::alloc::handle_alloc_error(layout));

        let new_base = unsafe { alloc.cast::<Value<'ctx>>().add(new_array_cap) };
        let src_base = unsafe { self.raw.sub(self.array_cap) };
        let dest_base = unsafe { new_base.sub(self.array_cap) };

        let copy_len = self.array_cap + 2 * self.map_cap;

        unsafe {
            core::ptr::copy_nonoverlapping(src_base.as_ptr(), dest_base.as_ptr(), copy_len);
        }

        let old_total_cap = 2 * self.map_cap + self.array_cap;

        let old_layout = Layout::array::<Value<'ctx>>(old_total_cap).unwrap();

        unsafe {
            self.arena.deallocate(src_base.cast(), old_layout);
        }

        self.raw = new_base;
        self.array_cap = new_array_cap;
        self.map_cap = new_map_cap;
    }

    fn grow_array(&mut self, i: usize) {
        self.raw_grow(self.map_cap, i.next_multiple_of(32));
    }

    fn grow_map(&mut self) {
        // TODO: We can optimize this in particular, just do a raw_grow for now
        self.raw_grow((self.map_cap + 1).next_power_of_two(), self.array_cap);
    }

    fn hash_one<'a>(&'a self, engine: &'ctx LuaEngine<'ctx>, v: Value<'ctx>) -> u64 {
        let mut state = self.hasher.build_hasher();
        engine.raw_hash(v, &mut state);
        state.finish()
    }
    pub fn insert(&mut self, engine: &'ctx LuaEngine<'ctx>, key: Value<'ctx>, value: Value<'ctx>) {
        if let Some(i) = key
            .as_int()
            .and_then(|x| usize::try_from(x).ok())
            .and_then(|x| isize::try_from(x).ok())
        {
            if (i as usize) > self.array_cap {
                self.grow_array(i as usize);
            }
            let offset = -i;

            let ptr = unsafe { self.raw.offset(offset) };
            unsafe {
                ptr.write(value);
            }
            if !value.is_nil() {
                self.slen = self.slen.max(i as usize);
            }
        } else {
            let hash = self.hash_one(engine, key);
            let data = self
                .htb
                .find(hash, |&v| {
                    engine.raw_eq_lookup(unsafe { self.raw.add(2 * v).read() }, value)
                })
                .copied();

            if let Some(data) = data {
                unsafe { self.raw.add(2 * data + 1).write(value) };
            } else {
                let mut n = self.next_map_slot;
                let mut found = false;
                let mut index = 0;

                while n < self.map_cap {
                    let key = unsafe { self.raw.add(2 * n).read() };

                    if let Type::Managed(ManagedType::Dead | ManagedType::NullTy) = key.type_of() {
                        if found {
                            break;
                        } else {
                            found = true;
                            index = n;
                            unsafe {
                                self.raw.add(2 * n).cast::<[Value; 2]>().write([key, value]);
                            }
                        }
                    }
                    n += 1;
                }

                if !found {
                    self.grow_map();
                    unsafe {
                        self.raw.add(2 * n).cast::<[Value; 2]>().write([key, value]);
                    }
                    index = n;
                    n += 1;
                }
                self.next_map_slot = n;

                self.htb.insert_unique(hash, index, |&v| {
                    let val = unsafe { self.raw.add(2 * v).read() };
                    let mut hasher = self.hasher.build_hasher();
                    engine.raw_hash(val, &mut hasher);
                    hasher.finish()
                });
            }
        }
    }

    pub fn get(&self, engine: &'ctx LuaEngine<'ctx>, key: Value<'ctx>) -> Option<Value<'ctx>> {
        if let Some(i) = key
            .as_int()
            .and_then(|x| usize::try_from(x).ok())
            .and_then(|x| isize::try_from(x).ok())
        {
            if (i as usize) < self.array_cap {
                Some(unsafe { self.raw.offset(-i).read().normalize() })
            } else {
                None
            }
        } else {
            let hash = self.hash_one(engine, key);
            if let Some(v) = self
                .htb
                .find(hash, |&v| {
                    engine.raw_eq_lookup(unsafe { self.raw.add(2 * v).read() }, key)
                })
                .copied()
            {
                Some(unsafe { self.raw.add(2 * v + 1).read().normalize() })
            } else {
                None
            }
        }
    }

    pub fn new(engine: &'ctx LuaEngine<'ctx>) -> Self {
        let layout = Layout::array::<Value<'ctx>>(2 * INIT_MAP_CAP + INIT_ARRAY_CAP).unwrap();
        let raw = unsafe {
            engine
                .alloc()
                .allocate_zeroed(layout)
                .unwrap_or_else(|_| alloc::alloc::handle_alloc_error(layout))
                .cast::<Value<'ctx>>()
                .add(INIT_ARRAY_CAP)
        };

        Self {
            slen: 0,
            htb: HashTable::with_capacity_in(INIT_MAP_CAP, engine.alloc()),
            raw,
            array_cap: INIT_ARRAY_CAP,
            map_cap: INIT_MAP_CAP,
            next_map_slot: 0,
            metatable: ArenaPtr::null(),
            gc_behaviour: TableGcDisposition {
                keys_weak: false,
                values_weak: false,
                needs_finalize_call: false,
            },
            hasher: RandomState::new(),
            arena: engine.alloc(),
        }
    }
}

const INIT_ARRAY_CAP: usize = 16;
const INIT_MAP_CAP: usize = 16;

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct TableGcDisposition {
    pub keys_weak: bool,
    pub values_weak: bool,
    pub needs_finalize_call: bool,
}
