use super::ffi;
use crate::error::LocalError;
use crate::model::GenParams;
use std::ffi::CString;
use std::ptr::NonNull;

/// An owned shim sampler chain, freed on drop.
/// Built from `GenParams` after checking temperature, top-p and the
/// grammar strings; one sampler serves one generation.
pub(crate) struct Sampler(NonNull<ffi::TkSampler>);

impl Sampler {
    /// Builds the chain: grammar, end-of-text ban, then top-p,
    /// temperature and seed (or greedy at temperature 0).
    /// Output: the sampler, or `Grammar` when llama.cpp refuses the GBNF.
    pub(crate) fn new(
        model: NonNull<ffi::TkModel>,
        params: &GenParams,
    ) -> Result<Self, LocalError> {
        let temp = params.temperature();
        if !temp.is_finite() || temp < 0.0 {
            return Err(LocalError::Decode(format!(
                "temperature {temp} is not a finite value at least 0"
            )));
        }
        let top_p = params.top_p();
        if !top_p.is_finite() || top_p <= 0.0 || top_p > 1.0 {
            return Err(LocalError::Decode(format!(
                "top_p {top_p} is outside (0, 1]"
            )));
        }
        let grammar = match params.grammar() {
            Some(g) => Some((
                CString::new(g.gbnf())
                    .map_err(|_| LocalError::Grammar("grammar holds a NUL byte".into()))?,
                CString::new(g.root())
                    .map_err(|_| LocalError::Grammar("root holds a NUL byte".into()))?,
            )),
            None => None,
        };
        let (gbnf, root) = match &grammar {
            Some((g, r)) => (g.as_ptr(), r.as_ptr()),
            None => (std::ptr::null(), std::ptr::null()),
        };
        let raw = unsafe {
            ffi::tk_sampler_new(
                model.as_ptr(),
                temp,
                top_p,
                params.seed(),
                i32::from(params.ban_eos()),
                gbnf,
                root,
            )
        };
        match NonNull::new(raw) {
            Some(ptr) => Ok(Self(ptr)),
            None if grammar.is_some() => Err(LocalError::Grammar(
                "llama.cpp could not parse the grammar".into(),
            )),
            None => Err(LocalError::Decode(
                "llama.cpp could not build the sampler".into(),
            )),
        }
    }

    /// Samples and accepts one token from the last output row.
    /// Input: the context of the last decode.
    /// Output: the token id as llama.cpp returns it.
    pub(crate) fn sample(&mut self, ctx: NonNull<ffi::TkCtx>) -> i32 {
        unsafe { ffi::tk_sample(self.0.as_ptr(), ctx.as_ptr(), -1) }
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        unsafe { ffi::tk_sampler_free(self.0.as_ptr()) }
    }
}
