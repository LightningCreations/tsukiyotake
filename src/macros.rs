macro_rules! nz {
    ($lit:literal) => {
        const {
            let __x = $lit;

            assert!(__x != 0);

            unsafe { ::core::num::NonZero::new_unchecked(__x) }
        }
    };
}

macro_rules! wtry_cf {
    ($expr:expr) => {
        match ::core::ops::Try::branch($expr) {
            ::core::ops::ControlFlow::Continue(__c) => __c,
            ::core::ops::ControlFlow::Break(__b) => {
                return ::core::ops::ControlFlow::Break(::core::ops::FromResidual::from_residual(
                    __b,
                ))
            }
        }
    };
}
