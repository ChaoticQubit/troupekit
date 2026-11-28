#ifndef TK_LLAMA_H
#define TK_LLAMA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// A loaded model with its vocabulary and end-of-generation token list.
/// Opaque to Rust; created by tk_model_load, released by tk_model_free.
/// Only this shim sees llama.cpp structs.
typedef struct tk_model tk_model;

/// One inference context with its token batch.
/// Opaque to Rust; created by tk_ctx_new, released by tk_ctx_free.
/// Not thread safe: Rust holds it behind a mutex.
typedef struct tk_ctx tk_ctx;

/// A sampler chain: grammar, logit bias, top-p, temperature, seed.
/// Opaque to Rust; created by tk_sampler_new, released by tk_sampler_free.
/// Holds grammar state, so one sampler serves one generation.
typedef struct tk_sampler tk_sampler;

/// Initializes the llama.cpp backends and silences its log output.
/// Input: none. Output: none.
/// Rust calls it once, guarded by a std::sync::Once.
void tk_backend_init(void);

/// Returns the backend and CPU feature summary of llama.cpp.
/// Input: none. Output: a static NUL-terminated string.
/// The caller never frees it.
const char *tk_system_info(void);

/// Loads a GGUF model from a NUL-terminated path.
/// Inputs: path, GPU layers (negative for all), mmap flag (0 or 1).
/// Output: the model, or NULL when llama.cpp refuses the file.
tk_model *tk_model_load(const char *path, int32_t gpu_layers, int32_t mmap);

/// Releases a model and its token lists.
/// Input: a model from tk_model_load, or NULL.
/// Output: none. Every context of the model must be freed first.
void tk_model_free(tk_model *model);

/// Returns the context length the model was trained with.
/// Input: a model. Output: tokens, above 0 for a valid model.
/// tk_ctx_new may use a smaller context.
int32_t tk_model_n_ctx_train(const tk_model *model);

/// Returns the vocabulary size.
/// Input: a model. Output: the token count.
/// Every token id is in 0 to this value minus 1.
int32_t tk_model_n_vocab(const tk_model *model);

/// Returns the embedding size of the model.
/// Input: a model. Output: floats per embedding vector.
/// tk_embeddings returns this many floats.
int32_t tk_model_n_embd(const tk_model *model);

/// Returns the length of each embedding output row.
/// Input: a model. Output: floats per row of tk_embeddings.
/// May differ from tk_model_n_embd for models with an output projection.
int32_t tk_model_n_embd_out(const tk_model *model);

/// Reports whether the model carries a default chat template.
/// Input: a model. Output: 1 when present, 0 otherwise.
/// Without one tk_chat_template_apply returns -1.
int32_t tk_model_has_template(const tk_model *model);

/// Creates a context for one sequence, with pooling off so embeddings
/// are per-token rows. Inputs: model, context and batch size in tokens,
/// threads, and the initial embedding flag. Output: the context, or NULL.
tk_ctx *tk_ctx_new(tk_model *model, uint32_t n_ctx, uint32_t n_batch, int32_t threads, int32_t embeddings);

/// Releases a context and its batch.
/// Input: a context from tk_ctx_new, or NULL.
/// Output: none.
void tk_ctx_free(tk_ctx *ctx);

/// Returns the context length of a context.
/// Input: a context. Output: tokens.
/// Prompt plus generated tokens stay below it.
uint32_t tk_ctx_n_ctx(const tk_ctx *ctx);

/// Returns the most tokens one tk_decode_batch call accepts.
/// Input: a context. Output: tokens.
/// Equal to the n_batch given to tk_ctx_new.
uint32_t tk_ctx_n_batch(const tk_ctx *ctx);

/// Turns text into token ids.
/// Inputs: model, text and its byte length, output buffer and capacity,
/// add-special and parse-special flags. Output: the count, or minus the
/// count needed when the buffer is short, or INT32_MIN on overflow.
int32_t tk_tokenize(const tk_model *model, const char *text, int32_t text_len, int32_t *out, int32_t cap,
                    int32_t add_special, int32_t parse_special);

/// Turns token ids into text, special tokens not rendered.
/// Inputs: model, ids and count, output buffer and capacity in bytes.
/// Output: bytes written, or minus the bytes needed when short.
int32_t tk_detokenize(const tk_model *model, const int32_t *tokens, int32_t n_tokens, char *out, int32_t cap);

/// Writes the text of one token, without a NUL.
/// Inputs: model, token id, output buffer and capacity in bytes.
/// Output: bytes written, or minus the bytes needed when short.
int32_t tk_token_piece(const tk_model *model, int32_t token, char *out, int32_t cap);

/// Reports whether a token ends generation.
/// Inputs: model and token id.
/// Output: 1 for an end-of-generation token, 0 otherwise.
int32_t tk_is_eog(const tk_model *model, int32_t token);

/// Renders messages with the model's default chat template.
/// Inputs: model, parallel arrays of NUL-terminated roles and contents,
/// count, add-assistant flag, output buffer and capacity. Output: the
/// full length (may exceed capacity), or -1 when the template is unknown.
int32_t tk_chat_template_apply(const tk_model *model, const char *const *roles, const char *const *contents,
                               size_t n_msg, int32_t add_assistant, char *out, int32_t cap);

/// Decodes tokens at positions pos0 onwards in sequence 0.
/// Inputs: context, ids and count (1 to n_batch), first position, and
/// all-logits flag (0: last token only). Output: llama_decode's code.
int32_t tk_decode_batch(tk_ctx *ctx, const int32_t *tokens, int32_t n_tokens, int32_t pos0, int32_t all_logits);

/// Builds a sampler chain.
/// Inputs: model, temperature (0 or less is greedy), top-p, seed,
/// ban-eos flag, and NUL-terminated GBNF and root rule (both NULL for
/// none). Output: the sampler, or NULL when the grammar does not parse.
tk_sampler *tk_sampler_new(const tk_model *model, float temp, float top_p, uint32_t seed, int32_t ban_eos,
                           const char *grammar, const char *root);

/// Releases a sampler chain and its samplers.
/// Input: a sampler from tk_sampler_new, or NULL.
/// Output: none.
void tk_sampler_free(tk_sampler *sampler);

/// Samples and accepts one token from an output row of the last decode.
/// Inputs: sampler, context, output index (-1 for the last).
/// Output: the token id.
int32_t tk_sample(tk_sampler *sampler, tk_ctx *ctx, int32_t idx);

/// Returns the logits of an output row of the last decode.
/// Inputs: context and output index (-1 for the last).
/// Output: n_vocab floats owned by the context, or NULL.
const float *tk_logits(tk_ctx *ctx, int32_t idx);

/// Returns the embedding of an output row of the last decode.
/// Inputs: context and output index. Output: n_embd floats owned by the
/// context (tk_model_n_embd_out of them), or NULL when output is off.
const float *tk_embeddings(tk_ctx *ctx, int32_t idx);

/// Switches embedding output on or off for the next decodes.
/// Inputs: context and flag (0 or 1).
/// Output: none.
void tk_set_embeddings(tk_ctx *ctx, int32_t on);

/// Drops every cached token of the context.
/// Input: a context.
/// Output: none.
void tk_kv_clear(tk_ctx *ctx);

/// Drops cached tokens from position n_keep onwards.
/// Inputs: context and positions to keep. Output: 1 on success, 0 when
/// the architecture cannot drop part of a sequence (clear it instead).
int32_t tk_kv_trim(tk_ctx *ctx, int32_t n_keep);

/// Reads the performance counters of the context.
/// Inputs: context and four out-pointers: prompt ms, generation ms,
/// prompt tokens, generated tokens. Output: none.
void tk_perf(const tk_ctx *ctx, double *prompt_ms, double *eval_ms, int32_t *n_prompt, int32_t *n_eval);

/// Resets the performance counters of the context.
/// Input: a context.
/// Output: none.
void tk_perf_reset(tk_ctx *ctx);

#ifdef __cplusplus
}
#endif

#endif
