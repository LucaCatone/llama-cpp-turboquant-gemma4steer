#pragma once

#include "llama.h"

#include "ggml-cpp.h"

#include <string>
#include <unordered_map>
#include <vector>

// TODO: pimpl

//
// llama_adapter_cvec
//

struct llama_adapter_cvec {
    ggml_tensor * tensor_for(int il) const;

    ggml_tensor * apply_to(ggml_context * ctx, ggml_tensor * cur, int  il) const;

    bool apply(
            const llama_model & model,
            const float * data,
            size_t len,
            int32_t n_embd,
            int32_t il_start,
            int32_t il_end);

private:
    bool init(const llama_model & model);

    int32_t layer_start = -1;
    int32_t layer_end   = -1;

    std::vector<ggml_context_ptr> ctxs;
    std::vector<ggml_backend_buffer_ptr> bufs;

    std::vector<ggml_tensor *> tensors; // per layer
};

using llama_adapter_cvec_ptr = std::shared_ptr<llama_adapter_cvec>;

//
// llama_adapter_lora
//

struct llama_adapter_lora_weight {
    ggml_tensor * a = nullptr;
    ggml_tensor * b = nullptr;

    // get actual scale based on rank and alpha
    float get_scale(float alpha, float adapter_scale) const {
        const float rank  = (float) b->ne[0];
        const float scale = alpha ? adapter_scale * alpha / rank : adapter_scale;
        return scale;
    }

    llama_adapter_lora_weight() = default;
    llama_adapter_lora_weight(ggml_tensor * a, ggml_tensor * b) : a(a), b(b) {}
};

struct llama_adapter_lora {
    llama_model * model = nullptr;

    // map tensor name to lora_a_b
    std::unordered_map<std::string, llama_adapter_lora_weight> ab_map;

    std::vector<ggml_context_ptr> ctxs;
    std::vector<ggml_backend_buffer_ptr> bufs;

    float alpha;

    // gguf metadata
    std::unordered_map<std::string, std::string> gguf_kv;

    // activated lora (aLoRA)
    std::vector<llama_token> alora_invocation_tokens;

    explicit llama_adapter_lora(llama_model * model) : model(model) {}
    ~llama_adapter_lora() = default;

    llama_adapter_lora_weight * get_weight(ggml_tensor * w);

    uint32_t get_n_nodes() const {
        return ab_map.size() * 6u; // a, b, scale, add, 2 x mul_mat
    }
};

using llama_adapter_loras = std::unordered_map<llama_adapter_lora *, float>;
using llama_adapter_loras_ptr = std::unique_ptr<llama_adapter_loras>;

//
// llama_kv_bank
//

struct llama_kv_bank {
    struct bank_layer {
        int32_t il = 0;
        int32_t n_slots = 0;

        // flat float data: [K: n_embd_head * n_kv_heads * n_slots]
        //                    [V: n_embd_head * n_kv_heads * n_slots]
        // stored pre-RoPE, pre-WHT (raw float32) by default.
        // Set already_rotated=true when extracted from cache (post-RoPE)
        // to skip the rotate transform in build_kv_bank_injection.
        std::vector<float> k_data;
        std::vector<float> v_data;
        bool already_rotated = false;

        // ggml tensors (pre-allocated, one per bank load)
        // Created in set_kv_bank, consumed by build_kv_bank_injection
        ggml_tensor * k_tensor = nullptr;
        ggml_tensor * v_tensor = nullptr;
    };

    int32_t n_embd_head = 0;  // head_dim del modello (originale, non padded)
    int32_t n_kv_heads  = 0;  // numero di KV heads

    std::vector<bank_layer> layers;

    // ggml context holding the pre-allocated tensors
    ggml_context * ctx = nullptr;

    bool has_layer(int il) const {
        for (auto & l : layers) {
            if (l.il == il) return true;
        }
        return false;
    }
};

using llama_kv_bank_ptr = std::shared_ptr<llama_kv_bank>;
