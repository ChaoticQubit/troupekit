#include "tk_llama.h"

#include <stdbool.h>
#include <stdlib.h>

#include "llama.h"

struct tk_model {
    struct llama_model *model;
    const struct llama_vocab *vocab;
    int32_t n_vocab;
    llama_logit_bias *eog_bias;
    int32_t n_eog;
};

struct tk_ctx {
    struct llama_context *ctx;
    struct llama_batch batch;
    int32_t n_batch;
};

struct tk_sampler {
    struct llama_sampler *chain;
};

static void tk_log_silent(enum ggml_log_level level, const char *text, void *user_data) {
    (void)level;
    (void)text;
    (void)user_data;
}

void tk_backend_init(void) {
    llama_log_set(tk_log_silent, NULL);
    llama_backend_init();
}

const char *tk_system_info(void) {
    return llama_print_system_info();
}

tk_model *tk_model_load(const char *path, int32_t gpu_layers, int32_t mmap) {
    if (path == NULL) {
        return NULL;
    }
    struct llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = gpu_layers;
    params.load_mode = mmap ? LLAMA_LOAD_MODE_MMAP : LLAMA_LOAD_MODE_NONE;
    struct llama_model *model = llama_model_load_from_file(path, params);
    if (model == NULL) {
        return NULL;
    }
    tk_model *out = calloc(1, sizeof(tk_model));
    if (out == NULL) {
        llama_model_free(model);
        return NULL;
    }
    out->model = model;
    out->vocab = llama_model_get_vocab(model);
    out->n_vocab = llama_vocab_n_tokens(out->vocab);
    int32_t n_eog = 0;
    for (int32_t t = 0; t < out->n_vocab; t++) {
        if (llama_vocab_is_eog(out->vocab, t)) {
            n_eog++;
        }
    }
    if (n_eog > 0) {
        out->eog_bias = calloc((size_t)n_eog, sizeof(llama_logit_bias));
        if (out->eog_bias == NULL) {
            tk_model_free(out);
            return NULL;
        }
        int32_t k = 0;
        for (int32_t t = 0; t < out->n_vocab && k < n_eog; t++) {
            if (llama_vocab_is_eog(out->vocab, t)) {
                out->eog_bias[k].token = t;
                out->eog_bias[k].bias = -1.0e30f;
                k++;
            }
        }
        out->n_eog = k;
    }
    return out;
}

void tk_model_free(tk_model *model) {
    if (model == NULL) {
        return;
    }
    free(model->eog_bias);
    llama_model_free(model->model);
    free(model);
}

int32_t tk_model_n_ctx_train(const tk_model *model) {
    return llama_model_n_ctx_train(model->model);
}

int32_t tk_model_n_vocab(const tk_model *model) {
    return model->n_vocab;
}

int32_t tk_model_n_embd(const tk_model *model) {
    return llama_model_n_embd(model->model);
}

int32_t tk_model_n_embd_out(const tk_model *model) {
    return llama_model_n_embd_out(model->model);
}

int32_t tk_model_has_template(const tk_model *model) {
    return llama_model_chat_template(model->model, NULL) != NULL ? 1 : 0;
}

tk_ctx *tk_ctx_new(tk_model *model, uint32_t n_ctx, uint32_t n_batch, int32_t threads, int32_t embeddings) {
    if (model == NULL || n_ctx == 0 || n_batch == 0 || n_batch > n_ctx || threads < 1) {
        return NULL;
    }
    struct llama_context_params params = llama_context_default_params();
    params.n_ctx = n_ctx;
    params.n_batch = n_batch;
    params.n_ubatch = n_batch;
    params.n_seq_max = 1;
    params.n_threads = threads;
    params.n_threads_batch = threads;
    params.embeddings = embeddings != 0;
    params.no_perf = false;
    struct llama_context *ctx = llama_init_from_model(model->model, params);
    if (ctx == NULL) {
        return NULL;
    }
    tk_ctx *out = calloc(1, sizeof(tk_ctx));
    if (out == NULL) {
        llama_free(ctx);
        return NULL;
    }
    out->ctx = ctx;
    out->batch = llama_batch_init((int32_t)n_batch, 0, 1);
    out->n_batch = (int32_t)n_batch;
    return out;
}

void tk_ctx_free(tk_ctx *ctx) {
    if (ctx == NULL) {
        return;
    }
    llama_batch_free(ctx->batch);
    llama_free(ctx->ctx);
    free(ctx);
}

uint32_t tk_ctx_n_ctx(const tk_ctx *ctx) {
    return llama_n_ctx(ctx->ctx);
}

uint32_t tk_ctx_n_batch(const tk_ctx *ctx) {
    return (uint32_t)ctx->n_batch;
}

int32_t tk_tokenize(const tk_model *model, const char *text, int32_t text_len, int32_t *out, int32_t cap,
                    int32_t add_special, int32_t parse_special) {
    return llama_tokenize(model->vocab, text, text_len, out, cap, add_special != 0, parse_special != 0);
}

int32_t tk_detokenize(const tk_model *model, const int32_t *tokens, int32_t n_tokens, char *out, int32_t cap) {
    return llama_detokenize(model->vocab, tokens, n_tokens, out, cap, false, false);
}

int32_t tk_token_piece(const tk_model *model, int32_t token, char *out, int32_t cap) {
    if (token < 0 || token >= model->n_vocab) {
        return 0;
    }
    return llama_token_to_piece(model->vocab, token, out, cap, 0, false);
}

int32_t tk_is_eog(const tk_model *model, int32_t token) {
    return llama_vocab_is_eog(model->vocab, token) ? 1 : 0;
}

int32_t tk_chat_template_apply(const tk_model *model, const char *const *roles, const char *const *contents,
                               size_t n_msg, int32_t add_assistant, char *out, int32_t cap) {
    const char *tmpl = llama_model_chat_template(model->model, NULL);
    if (tmpl == NULL || n_msg == 0) {
        return -1;
    }
    llama_chat_message *msgs = calloc(n_msg, sizeof(llama_chat_message));
    if (msgs == NULL) {
        return -1;
    }
    for (size_t i = 0; i < n_msg; i++) {
        msgs[i].role = roles[i];
        msgs[i].content = contents[i];
    }
    int32_t res = llama_chat_apply_template(tmpl, msgs, n_msg, add_assistant != 0, out, cap);
    free(msgs);
    return res < 0 ? -1 : res;
}

int32_t tk_decode_batch(tk_ctx *ctx, const int32_t *tokens, int32_t n_tokens, int32_t pos0, int32_t all_logits) {
    if (n_tokens < 1 || n_tokens > ctx->n_batch || pos0 < 0) {
        return -1;
    }
    for (int32_t i = 0; i < n_tokens; i++) {
        ctx->batch.token[i] = tokens[i];
        ctx->batch.pos[i] = pos0 + i;
        ctx->batch.n_seq_id[i] = 1;
        ctx->batch.seq_id[i][0] = 0;
        ctx->batch.logits[i] = (int8_t)((all_logits != 0 || i == n_tokens - 1) ? 1 : 0);
    }
    ctx->batch.n_tokens = n_tokens;
    return llama_decode(ctx->ctx, ctx->batch);
}

tk_sampler *tk_sampler_new(const tk_model *model, float temp, float top_p, uint32_t seed, int32_t ban_eos,
                           const char *grammar, const char *root) {
    struct llama_sampler *chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (chain == NULL) {
        return NULL;
    }
    if (grammar != NULL) {
        if (root == NULL) {
            llama_sampler_free(chain);
            return NULL;
        }
        struct llama_sampler *g = llama_sampler_init_grammar(model->vocab, grammar, root);
        if (g == NULL) {
            llama_sampler_free(chain);
            return NULL;
        }
        llama_sampler_chain_add(chain, g);
    }
    if (ban_eos != 0 && model->n_eog > 0) {
        llama_sampler_chain_add(chain, llama_sampler_init_logit_bias(model->n_vocab, model->n_eog, model->eog_bias));
    }
    if (temp <= 0.0f) {
        llama_sampler_chain_add(chain, llama_sampler_init_greedy());
    } else {
        llama_sampler_chain_add(chain, llama_sampler_init_top_p(top_p, 1));
        llama_sampler_chain_add(chain, llama_sampler_init_temp(temp));
        llama_sampler_chain_add(chain, llama_sampler_init_dist(seed));
    }
    tk_sampler *out = calloc(1, sizeof(tk_sampler));
    if (out == NULL) {
        llama_sampler_free(chain);
        return NULL;
    }
    out->chain = chain;
    return out;
}

void tk_sampler_free(tk_sampler *sampler) {
    if (sampler == NULL) {
        return;
    }
    llama_sampler_free(sampler->chain);
    free(sampler);
}

int32_t tk_sample(tk_sampler *sampler, tk_ctx *ctx, int32_t idx) {
    return llama_sampler_sample(sampler->chain, ctx->ctx, idx);
}

const float *tk_logits(tk_ctx *ctx, int32_t idx) {
    return llama_get_logits_ith(ctx->ctx, idx);
}

const float *tk_embeddings(tk_ctx *ctx, int32_t idx) {
    return llama_get_embeddings_ith(ctx->ctx, idx);
}

void tk_set_embeddings(tk_ctx *ctx, int32_t on) {
    llama_set_embeddings(ctx->ctx, on != 0);
}

void tk_kv_clear(tk_ctx *ctx) {
    llama_memory_clear(llama_get_memory(ctx->ctx), true);
}

int32_t tk_kv_trim(tk_ctx *ctx, int32_t n_keep) {
    if (n_keep < 0) {
        return 0;
    }
    return llama_memory_seq_rm(llama_get_memory(ctx->ctx), 0, n_keep, -1) ? 1 : 0;
}

void tk_perf(const tk_ctx *ctx, double *prompt_ms, double *eval_ms, int32_t *n_prompt, int32_t *n_eval) {
    struct llama_perf_context_data data = llama_perf_context(ctx->ctx);
    *prompt_ms = data.t_p_eval_ms;
    *eval_ms = data.t_eval_ms;
    *n_prompt = data.n_p_eval;
    *n_eval = data.n_eval;
}

void tk_perf_reset(tk_ctx *ctx) {
    llama_perf_context_reset(ctx->ctx);
}
