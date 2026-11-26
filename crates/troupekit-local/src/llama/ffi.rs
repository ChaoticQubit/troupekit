use std::ffi::c_char;
use std::marker::{PhantomData, PhantomPinned};

/// Opaque handle to a shim model (`tk_model` in `shim/tk_llama.h`).
/// Only ever used behind a raw pointer from `tk_model_load`.
/// Not `Send` or `Sync` by itself; `LlamaModel` adds the guarantees.
#[repr(C)]
pub(crate) struct TkModel {
    _data: [u8; 0],
    _marker: PhantomData<(*mut u8, PhantomPinned)>,
}

/// Opaque handle to a shim context (`tk_ctx` in `shim/tk_llama.h`).
/// Only ever used behind a raw pointer from `tk_ctx_new`.
/// Not thread safe; callers hold it behind a mutex.
#[repr(C)]
pub(crate) struct TkCtx {
    _data: [u8; 0],
    _marker: PhantomData<(*mut u8, PhantomPinned)>,
}

/// Opaque handle to a shim sampler chain (`tk_sampler`).
/// Only ever used behind a raw pointer from `tk_sampler_new`.
/// One sampler serves one generation.
#[repr(C)]
pub(crate) struct TkSampler {
    _data: [u8; 0],
    _marker: PhantomData<(*mut u8, PhantomPinned)>,
}

extern "C" {
    pub(crate) fn tk_backend_init();
    pub(crate) fn tk_system_info() -> *const c_char;
    pub(crate) fn tk_model_load(path: *const c_char, gpu_layers: i32, mmap: i32) -> *mut TkModel;
    pub(crate) fn tk_model_free(model: *mut TkModel);
    pub(crate) fn tk_model_n_ctx_train(model: *const TkModel) -> i32;
    pub(crate) fn tk_model_n_vocab(model: *const TkModel) -> i32;
    pub(crate) fn tk_model_n_embd(model: *const TkModel) -> i32;
    pub(crate) fn tk_model_n_embd_out(model: *const TkModel) -> i32;
    pub(crate) fn tk_model_has_template(model: *const TkModel) -> i32;
    pub(crate) fn tk_ctx_new(
        model: *mut TkModel,
        n_ctx: u32,
        n_batch: u32,
        threads: i32,
        embeddings: i32,
    ) -> *mut TkCtx;
    pub(crate) fn tk_ctx_free(ctx: *mut TkCtx);
    pub(crate) fn tk_ctx_n_ctx(ctx: *const TkCtx) -> u32;
    pub(crate) fn tk_ctx_n_batch(ctx: *const TkCtx) -> u32;
    pub(crate) fn tk_tokenize(
        model: *const TkModel,
        text: *const c_char,
        text_len: i32,
        out: *mut i32,
        cap: i32,
        add_special: i32,
        parse_special: i32,
    ) -> i32;
    pub(crate) fn tk_detokenize(
        model: *const TkModel,
        tokens: *const i32,
        n_tokens: i32,
        out: *mut c_char,
        cap: i32,
    ) -> i32;
    pub(crate) fn tk_token_piece(
        model: *const TkModel,
        token: i32,
        out: *mut c_char,
        cap: i32,
    ) -> i32;
    pub(crate) fn tk_is_eog(model: *const TkModel, token: i32) -> i32;
    pub(crate) fn tk_chat_template_apply(
        model: *const TkModel,
        roles: *const *const c_char,
        contents: *const *const c_char,
        n_msg: usize,
        add_assistant: i32,
        out: *mut c_char,
        cap: i32,
    ) -> i32;
    pub(crate) fn tk_decode_batch(
        ctx: *mut TkCtx,
        tokens: *const i32,
        n_tokens: i32,
        pos0: i32,
        all_logits: i32,
    ) -> i32;
    pub(crate) fn tk_sampler_new(
        model: *const TkModel,
        temp: f32,
        top_p: f32,
        seed: u32,
        ban_eos: i32,
        grammar: *const c_char,
        root: *const c_char,
    ) -> *mut TkSampler;
    pub(crate) fn tk_sampler_free(sampler: *mut TkSampler);
    pub(crate) fn tk_sample(sampler: *mut TkSampler, ctx: *mut TkCtx, idx: i32) -> i32;
    pub(crate) fn tk_logits(ctx: *mut TkCtx, idx: i32) -> *const f32;
    pub(crate) fn tk_embeddings(ctx: *mut TkCtx, idx: i32) -> *const f32;
    pub(crate) fn tk_set_embeddings(ctx: *mut TkCtx, on: i32);
    pub(crate) fn tk_kv_clear(ctx: *mut TkCtx);
    pub(crate) fn tk_kv_trim(ctx: *mut TkCtx, n_keep: i32) -> i32;
    pub(crate) fn tk_perf(
        ctx: *const TkCtx,
        prompt_ms: *mut f64,
        eval_ms: *mut f64,
        n_prompt: *mut i32,
        n_eval: *mut i32,
    );
    pub(crate) fn tk_perf_reset(ctx: *mut TkCtx);
}
