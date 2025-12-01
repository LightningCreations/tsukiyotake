use lock_api::{GuardSend, RawRwLock};

const EXCLUSIVE: usize = 1 << (usize::BITS - 1);

#[cfg(target_has_atomic = "ptr")]
mod imp {
    use core::sync::atomic::{AtomicUsize, Ordering};

    use super::EXCLUSIVE;

    pub type LockWord = AtomicUsize;

    pub fn test_and_inc(x: &LockWord) -> bool {
        x.fetch_update(Ordering::Relaxed, Ordering::Acquire, |x| {
            if (x & EXCLUSIVE) != 0 {
                None
            } else {
                Some(x + 1)
            }
        })
        .is_ok()
    }

    pub fn test_and_lock(x: &LockWord) -> bool {
        x.compare_exchange(0, EXCLUSIVE, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
    }

    pub fn decrement(x: &LockWord) {
        x.fetch_sub(1, Ordering::Release);
    }
    pub fn unlock(x: &LockWord) {
        x.store(0, Ordering::Release)
    }
}

#[cfg(not(target_has_atomic = "ptr"))]
mod imp {
    use super::EXCLUSIVE;

    pub type LockWord = core::cell::Cell<usize>;

    pub fn test_and_inc(x: &LockWord) -> bool {
        let v = x.get();
        if (v & EXCLUSIVE) != 0 {
            return false;
        }

        x.set(v + 1);
        true
    }

    pub fn test_and_lock(x: &LockWord) -> bool {
        if x.get() != 0 {
            return false;
        }
        x.store(EXCLUSIVE);
        true
    }

    pub fn decrement(x: &LockWord) {
        x.set(x.get() - 1)
    }

    pub fn unlock(x: &LockWord) {
        x.set(0)
    }
}

pub struct RawLock(imp::LockWord);

// SAFETY:
// This is only needed on non-atomic platforms, which we assume are single-threaded
unsafe impl Sync for RawLock {}

unsafe impl RawRwLock for RawLock {
    const INIT: Self = Self(imp::LockWord::new(0));

    type GuardMarker = GuardSend;

    fn lock_shared(&self) {
        while !self.try_lock_shared() {
            core::hint::spin_loop();
        }
    }

    fn try_lock_shared(&self) -> bool {
        imp::test_and_inc(&self.0)
    }

    unsafe fn unlock_shared(&self) {
        imp::decrement(&self.0);
    }

    fn lock_exclusive(&self) {
        while !self.try_lock_exclusive() {
            core::hint::spin_loop();
        }
    }

    fn try_lock_exclusive(&self) -> bool {
        imp::test_and_lock(&self.0)
    }

    unsafe fn unlock_exclusive(&self) {
        imp::unlock(&self.0);
    }
}

pub type RwLock<T: ?Sized> = lock_api::RwLock<RawLock, T>;
