macro_rules! nz {
    ($lit:literal) => {
        const {
            let __x = $lit;

            assert!(__x != 0);

            unsafe { ::core::num::NonZero::new_unchecked(__x) }
        }
    };
}
