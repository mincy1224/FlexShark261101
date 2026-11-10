#include "fs_bert_cls.hpp"

#include <algorithm>
#include <cstring>
#include <type_traits>
#include <vector>

#include <shark/protocols/add.hpp>
#include <shark/protocols/down.hpp>
#include <shark/protocols/gelu.hpp>
#include <shark/protocols/init.hpp>
#include <shark/protocols/input.hpp>
#include <shark/protocols/layernorm.hpp>
#include <shark/protocols/matmul.hpp>
#include <shark/protocols/matmulpar.hpp>
#include <shark/protocols/mul.hpp>
#include <shark/protocols/softmax.hpp>
#include <shark/protocols/truncate.hpp>
#include <shark/protocols/up.hpp>

#include <shark/utils/assert.hpp>
#include <shark/utils/globals.hpp>
#include <shark/utils/timer.hpp>

using namespace shark;
using namespace shark::protocols;

namespace sp = shark::protocols;

struct fs::bert_cls::model
{
    u64 n_layers = 12;
    u64 n_heads = 12;
    u64 n_embd = 768;
    u64 n_interm = 3072;
    int logdivisor = 3; 

    std::vector<span<u64>> c_attn_w;
    std::vector<span<u64>> c_attn_b;

    std::vector<span<u64>> c_proj_w;
    std::vector<span<u64>> c_proj_b;

    std::vector<span<u64>> ffn_up_w;
    std::vector<span<u64>> ffn_up_b;

    std::vector<span<u64>> ffn_down_w;
    std::vector<span<u64>> ffn_down_b;

    std::vector<span<u64>> ln1_w;
    std::vector<span<u64>> ln1_b;
    std::vector<span<u64>> ln2_w;
    std::vector<span<u64>> ln2_b;

    span<u64> classifier_w;
    span<u64> classifier_b;
    u64 num_labels = 2;

    bool accuracy_test = false;

    model
    (
        u64 layers, 
        u64 heads, 
        u64 embd, 
        u64 interm, 
        u64 labels, 
        bool acc_test
    )
        : 
            n_layers(layers), 
            n_heads(heads), 
            n_embd(embd), 
            n_interm(interm),
            num_labels(labels), 
            accuracy_test(acc_test),
            c_attn_w(n_layers), 
            c_attn_b(n_layers), 
            c_proj_w(n_layers), 
            c_proj_b(n_layers),
            ffn_up_w(n_layers), 
            ffn_up_b(n_layers), 
            ffn_down_w(n_layers), 
            ffn_down_b(n_layers),
            ln1_w(n_layers), 
            ln1_b(n_layers), 
            ln2_w(n_layers), 
            ln2_b(n_layers)
    {
        for (int i = 0; i < n_layers; ++i)
        {
            c_attn_w[i] = span<u64>(n_embd * 3 * n_embd);
            c_attn_b[i] = span<u64>(n_embd * 3);
            c_proj_w[i] = span<u64>(n_embd * n_embd);
            c_proj_b[i] = span<u64>(n_embd);
            ffn_up_w[i] = span<u64>(n_embd * n_interm);
            ffn_up_b[i] = span<u64>(n_interm);
            ffn_down_w[i] = span<u64>(n_interm * n_embd);
            ffn_down_b[i] = span<u64>(n_embd);

            ln1_w[i] = span<u64>(n_embd);
            ln1_b[i] = span<u64>(n_embd);
            ln2_w[i] = span<u64>(n_embd);
            ln2_b[i] = span<u64>(n_embd);
        }

        if (acc_test)
        {
            classifier_w = span<u64>(n_embd * num_labels);
            classifier_b = span<u64>(num_labels);
        }
    }
};

namespace
{
    using bert_model = fs::bert_cls::model;

    template <typename T>
    span<T> bert_gelu(span<T> &x)
    {
        return gelu::call(x);
    }

    template <typename T>
    span<T> view(span<T> &x, int n_token, int n_peices, int idx)
    {
        always_assert(x.size() % (n_token * n_peices) == 0);
        int n_embd = x.size() / n_token;
        span<T> y(x.size() / n_peices);

        #pragma omp parallel for collapse(2)
        for (int i = 0; i < n_token; i++)
        {
            for (int j = 0; j < (n_embd / n_peices); j++)
            {
                y[i * (n_embd / n_peices) + j] = x[i * n_embd + j + idx * (n_embd / n_peices)];
            }
        }

        return y;
    }

    span<u64> bert_softmax(u64 a, u64 b, span<u32> &x)
    {
        return softmax::call(a, b, x);
    }

    span<u64> linear(u64 a, u64 b, u64 c, span<u64> &x, const span<u64> &w, const span<u64> &bias)
    {
        auto res = matmul::call(a, b, c, x, w);
        res = truncate::call(res, FLOAT_PRECISION_64);
        res = add::call(res, bias);
        return res;
    }

    span<u16> linearDowncast16(u64 a, u64 b, u64 c, span<u64> &x, const span<u64> &w, const span<u64> &bias)
    {
        // Method 1: operation fusion
        auto res_tmp = matmul::call(a, b, c, x, w);
        res_tmp = add::call(res_tmp, bias);
        auto res = truncate::call_64_16(res_tmp, FLOAT_PRECISION_64);
        return res;
    }

    span<u64> ffn(span<u64> &x, int layer, bert_model &model)
    {
        u64 n_token = x.size() / model.n_embd;

        auto res_tmp = linearDowncast16(n_token, model.n_embd, model.n_interm, x, model.ffn_up_w[layer], model.ffn_up_b[layer]);

        utils::start_timer("nonlinear");
        res_tmp = bert_gelu(res_tmp);
        utils::stop_timer("nonlinear");

        utils::start_timer("up");
        auto res = up::call_16_64(res_tmp, FLOAT_PRECISION_16, FLOAT_PRECISION_64);
        utils::stop_timer("up");

        res = linear(n_token, model.n_interm, model.n_embd, res, model.ffn_down_w[layer], model.ffn_down_b[layer]);

        return res;
    }

    span<u64> bert_layernorm(span<u64> &x, u64 n_token, u64 n_embd, const span<u64> &gamma, const span<u64> &beta)
    {
        return layernorm::call(n_token, n_embd, x, gamma, beta);
    }

    span<u64> mha(span<u64> &x, int layer, bert_model &model, const span<u32> &mask)
    {
        u64 n_token = x.size() / model.n_embd;
        auto c = linear(n_token, model.n_embd, model.n_embd * 3, x, model.c_attn_w[layer], model.c_attn_b[layer]);

        auto q = view(c, n_token, 3, 0);
        auto k = view(c, n_token, 3, 1);
        auto v = view(c, n_token, 3, 2);

        // Batched MHA Optimization:
        u64 head_size = model.n_embd / model.n_heads;

        // 1. Compute Attention Scores (Q * K^T)
        span<u64> batched_q(model.n_heads * n_token * head_size);
        span<u64> batched_kt(model.n_heads * head_size * n_token);
        span<u64> batched_v(model.n_heads * n_token * head_size);

        // Optimized Q and V preparation
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < model.n_heads; i++)
        {
            for (int r = 0; r < n_token; ++r)
            {
                u64 dst_offset = i * n_token * head_size + r * head_size;
                u64 src_offset = r * model.n_embd + i * head_size;
                std::memcpy(batched_q.data() + dst_offset, q.data() + src_offset, head_size * sizeof(u64));
                std::memcpy(batched_v.data() + dst_offset, v.data() + src_offset, head_size * sizeof(u64));
            }
        }

        // K preparation requires transpose, so kept element-wise copy
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < model.n_heads; ++i)
        {
            for (int c = 0; c < head_size; ++c)
            {
                u64 *batched_kt_pointer = batched_kt.data() + i * head_size * n_token + c * n_token;
                int idx = i * head_size + c;
                for (int r = 0; r < n_token; ++r)
                {
                    batched_kt_pointer[r] = k[r * model.n_embd + idx];
                }
            }
        }

        // Batched MatMul: [n_heads, n_token, head_size] * [n_heads, head_size, n_token] -> [n_heads, n_token, n_token]
        auto batched_scores_raw = matmulpar::call(model.n_heads, n_token, head_size, n_token, batched_q, batched_kt);

        // Truncate (scale down by f + logdivisor)
        auto all_scores = truncate::call_64_32(batched_scores_raw, FLOAT_PRECISION_64 + model.logdivisor);

        // Apply Mask if present (Additive Masking)
        if (mask.data() != nullptr && mask.size() > 0)
        {
            #pragma omp parallel for collapse(3)
            for (int h = 0; h < model.n_heads; ++h)
            {
                for (int i = 0; i < n_token; ++i)
                {
                    for (int j = 0; j < n_token; ++j)
                    {
                        u64 idx = h * n_token * n_token + i * n_token + j;
                        all_scores[idx] = all_scores[idx] + mask[j];
                    }
                }
            }
        }

        // 2. Batched Softmax
        utils::start_timer("nonlinear");
        auto all_probs = bert_softmax(model.n_heads * n_token, n_token, all_scores);
        utils::stop_timer("nonlinear");

        // 3. Batched Prob * V
        span<u64> qks_sm_vs(n_token * model.n_embd);

        // Batched MatMul: Probs * V
        auto batched_res = matmulpar::call(model.n_heads, n_token, n_token, head_size, all_probs, batched_v);
        // Concat
        #pragma omp parallel for collapse(2)
        for (int r = 0; r < n_token; ++r)
        {
            for (int i = 0; i < model.n_heads; ++i)
            {
                u64 dst_offset = r * model.n_embd + i * head_size;
                u64 src_offset = i * n_token * head_size + r * head_size;
                std::memcpy(qks_sm_vs.data() + dst_offset, batched_res.data() + src_offset, head_size * sizeof(u64));
            }
        }

        c = truncate::call(qks_sm_vs, FLOAT_PRECISION_64);

        c = linear(n_token, model.n_embd, model.n_embd, c, model.c_proj_w[layer], model.c_proj_b[layer]);

        return c;
    }

    span<u64> layer(span<u64> &x, int layer, bert_model &model, const span<u32> &mask)
    {
        u64 n_token = x.size() / model.n_embd;
        // Post-LayerNorm Architecture (Standard BERT)
        // 1. Attention Sublayer
        utils::start_timer("mha");
        auto attn_out = mha(x, layer, model, mask);
        utils::stop_timer("mha");
        auto res = add::call(x, attn_out);
        utils::start_timer("layernorm");
        auto ln1 = bert_layernorm(res, n_token, model.n_embd, model.ln1_w[layer], model.ln1_b[layer]);
        utils::stop_timer("layernorm");

        // 2. FFN Sublayer
        utils::start_timer("ffn");
        auto ffn_out = ffn(ln1, layer, model);
        utils::stop_timer("ffn");
        res = add::call(ln1, ffn_out);
        utils::start_timer("layernorm");
        auto ln2 = bert_layernorm(res, n_token, model.n_embd, model.ln2_w[layer], model.ln2_b[layer]);
        utils::stop_timer("layernorm");

        return ln2;
    }

    span<u64> bert_classifier(span<u64> &hidden_states, bert_model &model)
    {
        // Extract [CLS] token (first token)
        span<u64> cls_token(model.n_embd);
        std::memcpy(cls_token.data(), hidden_states.data(), model.n_embd * sizeof(u64));

        // Classifier: Linear [1, n_embd] * [n_embd, num_labels] -> [1, num_labels]
        auto logits = linear(1, model.n_embd, model.num_labels, cls_token, model.classifier_w, model.classifier_b);

        return logits;
    }

    span<u64> inference(const span<u64> &x, bert_model &model, const span<u32> &mask)
    {
        span<u64> y = x;
        for (int i = 0; i < model.n_layers; i++)
        {
            y = layer(y, i, model, mask);
        }

        if (model.accuracy_test)
        {
            return bert_classifier(y, model);
        }
        return y;
    }

    void share_model_weights(bert_model &model, int owner)
    {
        for (int i = 0; i < model.n_layers; i++)
        {
            input::call(model.c_attn_w[i], owner);
            input::call(model.c_attn_b[i], owner);
            input::call(model.c_proj_w[i], owner);
            input::call(model.c_proj_b[i], owner);
            input::call(model.ffn_up_w[i], owner);
            input::call(model.ffn_up_b[i], owner);
            input::call(model.ffn_down_w[i], owner);
            input::call(model.ffn_down_b[i], owner);

            input::call(model.ln1_w[i], owner);
            input::call(model.ln1_b[i], owner);
            input::call(model.ln2_w[i], owner);
            input::call(model.ln2_b[i], owner);
        }

        if (model.accuracy_test)
        {
            input::call(model.classifier_w, owner);
            input::call(model.classifier_b, owner);
        }
    }
}

fs::bert_cls::bert_cls()
    : model_(std::make_unique<model>(NUM_LAYERS_, NUM_HEADS_, HIDDEN_SIZE_, FFN_SIZE_, NUM_LABELS_, true))
{
}

fs::bert_cls::~bert_cls() = default;

double fs::bert_cls::fix2double(shark::u64 x)
{
    return static_cast<double>(static_cast<std::make_signed_t<shark::u64>>(x)) / (1 << FLOAT_PRECISION_64);
}

void fs::bert_cls::load_(shark::span<shark::u64>& dst, std::span<const shark::u64> src)
{
    always_assert(src.size() == dst.size());
    std::copy(src.begin(), src.end(), dst.begin());
}

void fs::bert_cls::load_params(const params& p, int owner)
{
    always_assert(owner == sp::SERVER || owner == sp::CLIENT);
    if (sp::party == owner)
    {
        for (std::size_t i = 0; i < NUM_LAYERS_; ++i)
        {
            const auto& layer = p.layers[i];
            load_(model_->c_attn_w[i], layer.c_attn_w);
            load_(model_->c_attn_b[i], layer.c_attn_b);
            load_(model_->c_proj_w[i], layer.c_proj_w);
            load_(model_->c_proj_b[i], layer.c_proj_b);
            load_(model_->ffn_up_w[i], layer.ffn_up_w);
            load_(model_->ffn_up_b[i], layer.ffn_up_b);
            load_(model_->ffn_down_w[i], layer.ffn_down_w);
            load_(model_->ffn_down_b[i], layer.ffn_down_b);
            load_(model_->ln1_w[i], layer.ln1_w);
            load_(model_->ln1_b[i], layer.ln1_b);
            load_(model_->ln2_w[i], layer.ln2_w);
            load_(model_->ln2_b[i], layer.ln2_b);
        }
        load_(model_->classifier_w, p.classifier_w);
        load_(model_->classifier_b, p.classifier_b);
    }

    share_model_weights(*model_, owner);
    params_loaded_ = true;
}

shark::span<shark::u64> fs::bert_cls::inf(
    const shark::span<shark::u64>& input,
    const shark::span<shark::u32>& mask) const
{
    always_assert(params_loaded_);
    always_assert(input.size() > 0 && input.size() % HIDDEN_SIZE_ == 0);
    always_assert(mask.size() == 0 || mask.size() == input.size() / HIDDEN_SIZE_);
    return inference(input, *model_, mask);
}
