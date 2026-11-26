use super::ffi;
use super::metrics;
use super::sampler::Sampler;
use crate::error::LocalError;
use crate::model::{ChatMessage, Flow, GenParams, GenStop, Generation, LocalInfo, LocalModel};
use std::ffi::{c_char, CString};
use std::io::Read;
use std::path::Path;
use std::ptr::NonNull;
use std::sync::atomic::Ordering;
use std::sync::{Mutex, MutexGuard};
use std::time::Instant;

const CONTEXT_DEFAULT: usize = 4_096;
const THREADS_MAX: usize = 8;
const BATCH_MAX: usize = 512;
const FILE_MAX_BYTES: u64 = 64 * 1024 * 1024 * 1024;
const TEXT_MAX_BYTES: usize = 64 * 1024 * 1024;
const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const DECODE_NO_SLOT: i32 = 1;

/// Load settings for `LlamaModel`. Defaults: context 4 096 tokens, all
/// layers on the GPU with Metal or Vulkan and none on CPU, half the
/// cores (1 to 8) as threads, mmap on, embeddings off. `with_*` setters.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LlamaSettings {
    context: usize,
    gpu_layers: i32,
    threads: usize,
    mmap: bool,
    embeddings: bool,
}

impl Default for LlamaSettings {
    fn default() -> Self {
        let cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(2);
        let gpu = cfg!(any(
            target_os = "macos",
            target_os = "ios",
            feature = "llama-vulkan"
        ));
        Self {
            context: CONTEXT_DEFAULT,
            gpu_layers: if gpu { -1 } else { 0 },
            threads: (cores / 2).clamp(1, THREADS_MAX),
            mmap: true,
            embeddings: false,
        }
    }
}

impl LlamaSettings {
    /// Sets the context length in tokens; capped at the model's own.
    /// Valid from 16 upwards.
    /// Default 4 096.
    pub fn with_context(mut self, context: usize) -> Self {
        self.context = context;
        self
    }

    /// Sets how many layers go to the GPU; negative means all.
    /// 0 keeps the model on the CPU.
    /// Default all on Metal and Vulkan builds, 0 otherwise.
    pub fn with_gpu_layers(mut self, gpu_layers: i32) -> Self {
        self.gpu_layers = gpu_layers;
        self
    }

    /// Sets the inference threads.
    /// Valid from 1 to 256.
    /// Default half of the available cores, 1 to 8.
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.threads = threads;
        self
    }

    /// Sets whether the file is memory-mapped instead of read.
    /// Mapping keeps resident memory low.
    /// Default true.
    pub fn with_mmap(mut self, mmap: bool) -> Self {
        self.mmap = mmap;
        self
    }

    /// Lets `embed` run on this model.
    /// `LocalInfo::embedding_dims` is 0 unless this is set.
    /// Default false.
    pub fn with_embeddings(mut self, embeddings: bool) -> Self {
        self.embeddings = embeddings;
        self
    }

    /// Returns the requested context length in tokens.
    /// Input: none.
    /// Default 4 096.
    pub fn context(&self) -> usize {
        self.context
    }

    /// Returns the GPU layer count; negative means all.
    /// Input: none.
    /// Default all on Metal and Vulkan builds, 0 otherwise.
    pub fn gpu_layers(&self) -> i32 {
        self.gpu_layers
    }

    /// Returns the inference thread count.
    /// Input: none.
    /// Default half of the available cores, 1 to 8.
    pub fn threads(&self) -> usize {
        self.threads
    }

    /// Returns whether the file is memory-mapped.
    /// Input: none.
    /// Default true.
    pub fn mmap(&self) -> bool {
        self.mmap
    }

    /// Returns whether `embed` is allowed.
    /// Input: none.
    /// Default false.
    pub fn embeddings(&self) -> bool {
        self.embeddings
    }
}

struct State {
    ctx: NonNull<ffi::TkCtx>,
    cached: Vec<i32>,
}

/// A GGUF model run in-process by llama.cpp, checked before every C call.
/// Send and Sync: the model pointer is read-only after load and the
/// tokenizer API is thread safe (`llama.h:1265`); the one context is
/// used only under the mutex, which serialises generate, score, embed.
pub struct LlamaModel {
    model: NonNull<ffi::TkModel>,
    state: Mutex<State>,
    info: LocalInfo,
    n_ctx: usize,
    n_batch: usize,
    embeddings: bool,
}

unsafe impl Send for LlamaModel {}
unsafe impl Sync for LlamaModel {}

impl std::fmt::Debug for LlamaModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlamaModel")
            .field("info", &self.info)
            .finish_non_exhaustive()
    }
}

impl Drop for LlamaModel {
    fn drop(&mut self) {
        let state = self.state.get_mut().unwrap_or_else(|p| p.into_inner());
        unsafe {
            ffi::tk_ctx_free(state.ctx.as_ptr());
            ffi::tk_model_free(self.model.as_ptr());
        }
    }
}

impl LlamaModel {
    /// Loads a GGUF file. The path must be a regular file of at most
    /// 64 GiB that starts with the GGUF magic and holds no NUL byte.
    /// Output: the model, or `Load` naming the reason.
    pub fn load(path: &Path, settings: LlamaSettings) -> Result<Self, LocalError> {
        if settings.context < 16 {
            return Err(LocalError::Load(format!(
                "context {} is below 16 tokens",
                settings.context
            )));
        }
        if settings.threads == 0 || settings.threads > 256 {
            return Err(LocalError::Load(format!(
                "threads {} is outside 1 to 256",
                settings.threads
            )));
        }
        let c_path = path_to_cstring(path)?;
        check_gguf_file(path)?;
        super::init_backend();
        let raw = unsafe {
            ffi::tk_model_load(
                c_path.as_ptr(),
                settings.gpu_layers,
                i32::from(settings.mmap),
            )
        };
        let model = NonNull::new(raw)
            .ok_or_else(|| LocalError::Load(format!("llama.cpp refused {}", path.display())))?;
        let (train, vocab, embd, template) = unsafe {
            (
                ffi::tk_model_n_ctx_train(model.as_ptr()),
                ffi::tk_model_n_vocab(model.as_ptr()),
                ffi::tk_model_n_embd_out(model.as_ptr()),
                ffi::tk_model_has_template(model.as_ptr()),
            )
        };
        let free_model = |message: String| {
            unsafe { ffi::tk_model_free(model.as_ptr()) };
            LocalError::Load(message)
        };
        if train <= 0 || vocab <= 0 || embd <= 0 {
            return Err(free_model(format!(
                "model reports context {train}, vocabulary {vocab}, embedding {embd}"
            )));
        }
        let n_ctx = settings.context.min(train as usize);
        let n_batch = n_ctx.min(BATCH_MAX);
        let threads = i32::try_from(settings.threads).expect("threads checked at most 256");
        let ctx_raw = unsafe {
            ffi::tk_ctx_new(
                model.as_ptr(),
                u32::try_from(n_ctx).map_err(|_| free_model("context too large".into()))?,
                u32::try_from(n_batch).expect("batch at most 512"),
                threads,
                0,
            )
        };
        let Some(ctx) = NonNull::new(ctx_raw) else {
            return Err(free_model("llama.cpp could not create a context".into()));
        };
        let actual = unsafe { ffi::tk_ctx_n_ctx(ctx.as_ptr()) } as usize;
        let n_ctx = n_ctx.min(actual);
        let n_batch = n_batch.min(n_ctx);
        assert!(
            n_ctx > 0 && n_batch > 0,
            "context and batch hold at least one token"
        );
        let dims = if settings.embeddings {
            embd as usize
        } else {
            0
        };
        Ok(Self {
            model,
            state: Mutex::new(State {
                ctx,
                cached: Vec::new(),
            }),
            info: LocalInfo::new(n_ctx, vocab as usize, dims, template == 1),
            n_ctx,
            n_batch,
            embeddings: settings.embeddings,
        })
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        match self.state.lock() {
            Ok(guard) => guard,
            Err(poisoned) => {
                let mut guard = poisoned.into_inner();
                unsafe { ffi::tk_kv_clear(guard.ctx.as_ptr()) };
                guard.cached.clear();
                guard
            }
        }
    }

    fn check_tokens(&self, tokens: &[u32]) -> Result<Vec<i32>, LocalError> {
        let vocab = self.info.vocab_size();
        tokens
            .iter()
            .map(|t| {
                if (*t as usize) < vocab {
                    Ok(*t as i32)
                } else {
                    Err(LocalError::Tokenize(format!(
                        "token {t} is outside the vocabulary of {vocab}"
                    )))
                }
            })
            .collect()
    }

    fn piece(&self, token: i32) -> Vec<u8> {
        let mut buf = vec![0u8; 64];
        for _ in 0..2 {
            let cap = i32::try_from(buf.len()).expect("piece buffer is small");
            let n = unsafe {
                ffi::tk_token_piece(
                    self.model.as_ptr(),
                    token,
                    buf.as_mut_ptr().cast::<c_char>(),
                    cap,
                )
            };
            if n >= 0 {
                buf.truncate(n as usize);
                return buf;
            }
            buf.resize(n.unsigned_abs() as usize, 0);
        }
        Vec::new()
    }

    fn decode(
        &self,
        state: &mut State,
        tokens: &[i32],
        all_logits: bool,
    ) -> Result<(), LocalError> {
        for chunk in tokens.chunks(self.n_batch) {
            let pos = i32::try_from(state.cached.len()).expect("positions stay below the context");
            let n = i32::try_from(chunk.len()).expect("chunk at most 512");
            let rc = unsafe {
                ffi::tk_decode_batch(
                    state.ctx.as_ptr(),
                    chunk.as_ptr(),
                    n,
                    pos,
                    i32::from(all_logits),
                )
            };
            if rc != 0 {
                unsafe { ffi::tk_kv_clear(state.ctx.as_ptr()) };
                state.cached.clear();
                return Err(if rc == DECODE_NO_SLOT {
                    LocalError::ContextFull {
                        needed: pos as usize + chunk.len(),
                        limit: self.n_ctx,
                    }
                } else {
                    LocalError::Decode(format!("llama_decode returned {rc}"))
                });
            }
            state.cached.extend_from_slice(chunk);
        }
        Ok(())
    }

    fn reuse_prefix(&self, state: &mut State, prompt: &[i32]) -> usize {
        let mut keep = state
            .cached
            .iter()
            .zip(prompt)
            .take_while(|(a, b)| a == b)
            .count();
        if keep == prompt.len() {
            keep -= 1;
        }
        if keep < state.cached.len() {
            let trimmed = unsafe { ffi::tk_kv_trim(state.ctx.as_ptr(), keep as i32) } == 1;
            if trimmed {
                state.cached.truncate(keep);
            } else {
                unsafe { ffi::tk_kv_clear(state.ctx.as_ptr()) };
                state.cached.clear();
                keep = 0;
            }
        }
        assert_eq!(
            state.cached.len(),
            keep,
            "cache holds exactly the reused prefix"
        );
        keep
    }

    fn logits_row(&self, ctx: NonNull<ffi::TkCtx>, idx: i32) -> Result<&[f32], LocalError> {
        let ptr = unsafe { ffi::tk_logits(ctx.as_ptr(), idx) };
        if ptr.is_null() {
            return Err(LocalError::Decode(format!("no logits for output {idx}")));
        }
        Ok(unsafe { std::slice::from_raw_parts(ptr, self.info.vocab_size()) })
    }
}

impl LocalModel for LlamaModel {
    fn info(&self) -> &LocalInfo {
        &self.info
    }

    fn tokenize(&self, text: &str, special: bool) -> Result<Vec<u32>, LocalError> {
        if text.contains('\0') {
            return Err(LocalError::Tokenize("text holds a NUL byte".into()));
        }
        if text.len() > TEXT_MAX_BYTES {
            return Err(LocalError::Tokenize(format!(
                "text is {} bytes, limit {TEXT_MAX_BYTES}",
                text.len()
            )));
        }
        let len = i32::try_from(text.len()).expect("text at most 64 MiB");
        let mut out = vec![0i32; text.len() + 8];
        for _ in 0..2 {
            let cap = i32::try_from(out.len())
                .map_err(|_| LocalError::Tokenize("too many tokens".into()))?;
            let n = unsafe {
                ffi::tk_tokenize(
                    self.model.as_ptr(),
                    text.as_ptr().cast::<c_char>(),
                    len,
                    out.as_mut_ptr(),
                    cap,
                    1,
                    i32::from(special),
                )
            };
            if n == i32::MIN {
                return Err(LocalError::Tokenize("token count overflows".into()));
            }
            if n >= 0 {
                out.truncate(n as usize);
                return out
                    .iter()
                    .map(|t| {
                        u32::try_from(*t).map_err(|_| LocalError::Tokenize("negative token".into()))
                    })
                    .collect();
            }
            out.resize(n.unsigned_abs() as usize, 0);
        }
        Err(LocalError::Tokenize(
            "token buffer size did not settle".into(),
        ))
    }

    fn detokenize(&self, tokens: &[u32]) -> Result<String, LocalError> {
        let ids = self.check_tokens(tokens)?;
        if ids.is_empty() {
            return Ok(String::new());
        }
        let n_tokens =
            i32::try_from(ids.len()).map_err(|_| LocalError::Tokenize("too many tokens".into()))?;
        let mut buf = vec![0u8; ids.len().saturating_mul(8).saturating_add(16)];
        for _ in 0..2 {
            let cap = i32::try_from(buf.len())
                .map_err(|_| LocalError::Tokenize("text too large".into()))?;
            let n = unsafe {
                ffi::tk_detokenize(
                    self.model.as_ptr(),
                    ids.as_ptr(),
                    n_tokens,
                    buf.as_mut_ptr().cast::<c_char>(),
                    cap,
                )
            };
            if n >= 0 {
                buf.truncate(n as usize);
                return Ok(String::from_utf8_lossy(&buf).into_owned());
            }
            let need = n.unsigned_abs() as usize;
            if need > TEXT_MAX_BYTES {
                return Err(LocalError::Tokenize(format!(
                    "text would be {need} bytes, limit {TEXT_MAX_BYTES}"
                )));
            }
            buf.resize(need, 0);
        }
        Err(LocalError::Tokenize(
            "text buffer size did not settle".into(),
        ))
    }

    fn render_chat(&self, msgs: &[ChatMessage], add_assistant: bool) -> Result<String, LocalError> {
        if !self.info.chat_template() {
            return Err(LocalError::Unsupported("chat template"));
        }
        if msgs.is_empty() {
            return Err(LocalError::Tokenize("no messages to render".into()));
        }
        let total: usize = msgs.iter().map(|m| m.content().len() + 16).sum();
        if total > TEXT_MAX_BYTES {
            return Err(LocalError::Tokenize(format!(
                "messages are {total} bytes, limit {TEXT_MAX_BYTES}"
            )));
        }
        let to_c = |s: &str| {
            CString::new(s).map_err(|_| LocalError::Tokenize("message holds a NUL byte".into()))
        };
        let roles = msgs
            .iter()
            .map(|m| to_c(m.role().as_str()))
            .collect::<Result<Vec<_>, _>>()?;
        let contents = msgs
            .iter()
            .map(|m| to_c(m.content()))
            .collect::<Result<Vec<_>, _>>()?;
        let role_ptrs: Vec<*const c_char> = roles.iter().map(|c| c.as_ptr()).collect();
        let content_ptrs: Vec<*const c_char> = contents.iter().map(|c| c.as_ptr()).collect();
        let mut buf = vec![0u8; total.saturating_mul(2).saturating_add(256)];
        for _ in 0..2 {
            let cap = i32::try_from(buf.len())
                .map_err(|_| LocalError::Tokenize("prompt too large".into()))?;
            let n = unsafe {
                ffi::tk_chat_template_apply(
                    self.model.as_ptr(),
                    role_ptrs.as_ptr(),
                    content_ptrs.as_ptr(),
                    msgs.len(),
                    i32::from(add_assistant),
                    buf.as_mut_ptr().cast::<c_char>(),
                    cap,
                )
            };
            if n < 0 {
                return Err(LocalError::Unsupported("chat template"));
            }
            let n = n as usize;
            if n <= buf.len() {
                buf.truncate(n);
                return String::from_utf8(buf)
                    .map_err(|_| LocalError::Tokenize("template output is not UTF-8".into()));
            }
            if n > TEXT_MAX_BYTES {
                return Err(LocalError::Tokenize(format!(
                    "prompt would be {n} bytes, limit {TEXT_MAX_BYTES}"
                )));
            }
            buf.resize(n, 0);
        }
        Err(LocalError::Tokenize(
            "prompt buffer size did not settle".into(),
        ))
    }

    fn generate(
        &self,
        prompt: &[u32],
        params: &GenParams,
        on_token: &mut dyn FnMut(&str) -> Flow,
    ) -> Result<Generation, LocalError> {
        let started = Instant::now();
        let prompt = self.check_tokens(prompt)?;
        if prompt.is_empty() {
            return Err(LocalError::Tokenize("prompt is empty".into()));
        }
        if prompt.len() >= self.n_ctx {
            return Err(LocalError::ContextFull {
                needed: prompt.len() + 1,
                limit: self.n_ctx,
            });
        }
        let mut sampler = Sampler::new(self.model, params)?;
        let mut state = self.lock();
        let ctx = state.ctx;
        let reused = self.reuse_prefix(&mut state, &prompt);
        unsafe { ffi::tk_perf_reset(ctx.as_ptr()) };
        let cancelled = || params.cancel().is_some_and(|c| c.load(Ordering::SeqCst));
        for chunk in prompt[reused..].chunks(self.n_batch) {
            if cancelled() {
                let metrics = metrics::collect(ctx, 0.0, reused);
                return Ok(Generation::new(
                    String::new(),
                    0,
                    GenStop::Cancelled,
                    None,
                    metrics,
                ));
            }
            self.decode(&mut state, chunk, false)?;
        }
        let mut text = Gen::new(params.stop());
        let mut generated = 0usize;
        let mut logprob_sum = 0.0f64;
        let mut ttft_ms = 0.0f64;
        let mut stop = GenStop::MaxTokens;
        while generated < params.max_tokens() {
            if cancelled() {
                stop = GenStop::Cancelled;
                break;
            }
            if state.cached.len() >= self.n_ctx {
                return Err(LocalError::ContextFull {
                    needed: state.cached.len() + 1,
                    limit: self.n_ctx,
                });
            }
            let token = sampler.sample(ctx);
            if generated == 0 {
                ttft_ms = started.elapsed().as_secs_f64() * 1_000.0;
            }
            if token < 0 || token as usize >= self.info.vocab_size() {
                return Err(LocalError::Decode(format!(
                    "sampler returned token {token}"
                )));
            }
            if params.confidence() {
                logprob_sum += log_softmax_at(self.logits_row(ctx, -1)?, token as usize);
            }
            if !params.ban_eos() && unsafe { ffi::tk_is_eog(self.model.as_ptr(), token) } == 1 {
                stop = GenStop::EndOfText;
                break;
            }
            generated += 1;
            let step = text.push(&self.piece(token), on_token);
            if step != Flow::Continue {
                stop = GenStop::Stop;
                break;
            }
            self.decode(&mut state, &[token], false)?;
        }
        if stop != GenStop::Stop {
            text.finish(on_token);
        }
        let mean = (params.confidence() && generated > 0)
            .then(|| (logprob_sum / generated as f64).min(0.0))
            .filter(|m| m.is_finite());
        let metrics = metrics::collect(ctx, ttft_ms, reused);
        Ok(Generation::new(
            text.into_text(),
            generated,
            stop,
            mean,
            metrics,
        ))
    }

    fn score(&self, tokens: &[u32]) -> Result<f64, LocalError> {
        let ids = self.check_tokens(tokens)?;
        if ids.len() < 2 {
            return Ok(0.0);
        }
        if ids.len() > self.n_ctx {
            return Err(LocalError::ContextFull {
                needed: ids.len(),
                limit: self.n_ctx,
            });
        }
        let mut state = self.lock();
        let ctx = state.ctx;
        unsafe { ffi::tk_kv_clear(ctx.as_ptr()) };
        state.cached.clear();
        let mut sum = 0.0f64;
        let mut scored = 0usize;
        for chunk in ids.chunks(self.n_batch) {
            let base = state.cached.len();
            self.decode(&mut state, chunk, true)?;
            for i in 0..chunk.len() {
                let target = base + i + 1;
                if target < ids.len() {
                    let row = i32::try_from(i).expect("row inside a batch");
                    sum += log_softmax_at(self.logits_row(ctx, row)?, ids[target] as usize);
                    scored += 1;
                }
            }
        }
        assert_eq!(
            scored,
            ids.len() - 1,
            "every token after the first is scored"
        );
        Ok(sum.min(0.0))
    }

    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, LocalError> {
        if !self.embeddings {
            return Err(LocalError::Unsupported("embeddings"));
        }
        let token_lists = texts
            .iter()
            .map(|t| self.tokenize(t, false))
            .collect::<Result<Vec<_>, _>>()?;
        let dims = unsafe { ffi::tk_model_n_embd_out(self.model.as_ptr()) } as usize;
        let mut state = self.lock();
        let ctx = state.ctx;
        unsafe { ffi::tk_set_embeddings(ctx.as_ptr(), 1) };
        let result = token_lists
            .iter()
            .map(|tokens| self.embed_one(&mut state, tokens, dims))
            .collect();
        unsafe {
            ffi::tk_set_embeddings(ctx.as_ptr(), 0);
            ffi::tk_kv_clear(ctx.as_ptr());
        }
        state.cached.clear();
        result
    }

    fn reset(&self) {
        let mut state = self.lock();
        unsafe { ffi::tk_kv_clear(state.ctx.as_ptr()) };
        state.cached.clear();
    }
}

impl LlamaModel {
    fn embed_one(
        &self,
        state: &mut State,
        tokens: &[u32],
        dims: usize,
    ) -> Result<Vec<f32>, LocalError> {
        let ids = self.check_tokens(tokens)?;
        if ids.len() > self.n_ctx {
            return Err(LocalError::ContextFull {
                needed: ids.len(),
                limit: self.n_ctx,
            });
        }
        unsafe { ffi::tk_kv_clear(state.ctx.as_ptr()) };
        state.cached.clear();
        let mut sum = vec![0.0f32; dims];
        if ids.is_empty() {
            return Ok(sum);
        }
        for chunk in ids.chunks(self.n_batch) {
            self.decode(state, chunk, true)?;
            for i in 0..chunk.len() {
                let ptr = unsafe { ffi::tk_embeddings(state.ctx.as_ptr(), i as i32) };
                if ptr.is_null() {
                    return Err(LocalError::Decode(format!("no embedding for output {i}")));
                }
                let row = unsafe { std::slice::from_raw_parts(ptr, dims) };
                sum.iter_mut().zip(row).for_each(|(s, v)| *s += v);
            }
        }
        let norm = sum.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 && norm.is_finite() {
            sum.iter_mut().for_each(|x| *x /= norm);
        }
        Ok(sum)
    }
}

struct Gen<'a> {
    text: String,
    pending: Vec<u8>,
    emitted: usize,
    stops: &'a [String],
    hold: usize,
}

impl<'a> Gen<'a> {
    fn new(stops: &'a [String]) -> Self {
        let hold = stops
            .iter()
            .map(String::len)
            .max()
            .unwrap_or(0)
            .saturating_sub(1);
        Self {
            text: String::new(),
            pending: Vec::new(),
            emitted: 0,
            stops,
            hold,
        }
    }

    fn push(&mut self, bytes: &[u8], on_token: &mut dyn FnMut(&str) -> Flow) -> Flow {
        self.pending.extend_from_slice(bytes);
        match std::str::from_utf8(&self.pending) {
            Ok(s) => {
                self.text.push_str(s);
                self.pending.clear();
            }
            Err(e) if e.error_len().is_none() => {
                let valid = e.valid_up_to();
                let s = std::str::from_utf8(&self.pending[..valid]).expect("prefix is valid");
                self.text.push_str(s);
                self.pending.drain(..valid);
            }
            Err(_) => {
                self.text.push_str(&String::from_utf8_lossy(&self.pending));
                self.pending.clear();
            }
        }
        let mut search_from = self.emitted.saturating_sub(self.hold);
        while !self.text.is_char_boundary(search_from) {
            search_from -= 1;
        }
        let hit = self
            .stops
            .iter()
            .filter(|s| !s.is_empty())
            .filter_map(|s| {
                self.text[search_from..]
                    .find(s.as_str())
                    .map(|i| search_from + i)
            })
            .min();
        if let Some(at) = hit {
            self.text.truncate(at);
            self.emit_to(at, on_token);
            return Flow::Stop;
        }
        let mut safe = self.text.len().saturating_sub(self.hold);
        while !self.text.is_char_boundary(safe) {
            safe -= 1;
        }
        self.emit_to(safe, on_token)
    }

    fn emit_to(&mut self, end: usize, on_token: &mut dyn FnMut(&str) -> Flow) -> Flow {
        if end <= self.emitted {
            return Flow::Continue;
        }
        let flow = on_token(&self.text[self.emitted..end]);
        self.emitted = end;
        flow
    }

    fn finish(&mut self, on_token: &mut dyn FnMut(&str) -> Flow) {
        if !self.pending.is_empty() {
            self.text.push_str(&String::from_utf8_lossy(&self.pending));
            self.pending.clear();
        }
        let end = self.text.len();
        self.emit_to(end, on_token);
    }

    fn into_text(self) -> String {
        self.text
    }
}

fn log_softmax_at(logits: &[f32], token: usize) -> f64 {
    assert!(token < logits.len(), "token inside the logits row");
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let sum: f64 = logits.iter().map(|l| (*l as f64 - max).exp()).sum();
    (logits[token] as f64 - max) - sum.ln()
}

fn path_to_cstring(path: &Path) -> Result<CString, LocalError> {
    #[cfg(unix)]
    let bytes = {
        use std::os::unix::ffi::OsStrExt;
        path.as_os_str().as_bytes().to_vec()
    };
    #[cfg(not(unix))]
    let bytes = path
        .to_str()
        .ok_or_else(|| LocalError::Load(format!("path {} is not UTF-8", path.display())))?
        .as_bytes()
        .to_vec();
    CString::new(bytes).map_err(|_| LocalError::Load("model path holds a NUL byte".into()))
}

fn check_gguf_file(path: &Path) -> Result<(), LocalError> {
    let meta = std::fs::metadata(path)
        .map_err(|e| LocalError::Load(format!("{}: {e}", path.display())))?;
    if !meta.is_file() {
        return Err(LocalError::Load(format!(
            "{} is not a regular file",
            path.display()
        )));
    }
    if meta.len() > FILE_MAX_BYTES {
        return Err(LocalError::Load(format!(
            "{} is {} bytes, limit {FILE_MAX_BYTES}",
            path.display(),
            meta.len()
        )));
    }
    let mut magic = [0u8; 4];
    std::fs::File::open(path)
        .and_then(|mut f| f.read_exact(&mut magic))
        .map_err(|e| LocalError::Load(format!("{}: {e}", path.display())))?;
    if &magic != GGUF_MAGIC {
        return Err(LocalError::Load(format!(
            "{} is not a GGUF file",
            path.display()
        )));
    }
    Ok(())
}
