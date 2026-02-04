pub struct FmtWrapper<'a>(pub &'a mut [u8]);

impl<'a> core::fmt::Write for FmtWrapper<'a> {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        let len = s.len();

        if len > self.0.len() {
            return Err(core::fmt::Error);
        }

        replace_with::replace_with_or_abort(&mut self.0, |v| {
            let (l, r) = v.split_at_mut(len);
            l.copy_from_slice(s.as_bytes());
            r
        });

        Ok(())
    }
}
