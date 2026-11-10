#include "fs_bert_fv.hpp"

namespace fs
{        

    u64 double2fix(double x) 
    {
        return static_cast<u64>(static_cast<std::make_signed_t<u64>>(x * (1 << FLOAT_PRECISION_64)));  // Note: must truncate the decimal part using static_cast<int_t>
    }

    double fix2double(u64 x) 
    {
        return static_cast<double>(static_cast<std::make_signed_t<u64>>(x)) / (1 << FLOAT_PRECISION_64);
    }

    double fix2double(u32 x) 
    {
        return static_cast<double>(static_cast<std::make_signed_t<u32>>(x)) / (1 << FLOAT_PRECISION_32);
    }

    span<u64> bert_softmax(u64 a, u64 b, span<u32> &x)
    {
        // Use library call
        return softmax::call(a, b, x);
    }

    span<u64> linear(u64 a, u64 b, u64 c, span<u64> &x, const span<u64> &w, const span<u64> &bias)
    {
        auto res = matmul::call(a, b, c, x, w);
        res = truncate::call(res, FLOAT_PRECISION_64);
        res = add::call(res, bias);
        return res;
    }

    span<u32> linearDowncast32(u64 a, u64 b, u64 c, span<u64> &x, const span<u64> &w, const span<u64> &bias)
    {
        // Method 1: operation fusion
        auto res_tmp = matmul::call(a, b, c, x, w);
        res_tmp = add::call(res_tmp, bias);
        auto res = truncate::call_64_32(res_tmp, FLOAT_PRECISION_64);
        return res;

        // Method 2: do not double the precision of bias, extra one round
        // auto res_tmp = matmul::call(a, b, c, x, w);
        // res_tmp = truncate::call(res_tmp, FLOAT_PRECISION_64);
        // res_tmp = add::call(res_tmp, bias);
        // auto res = down::call_64_32(res_tmp, FLOAT_PRECISION_64, FLOAT_PRECISION_32);
        // return res;
    }

    span<u16> linearDowncast16(u64 a, u64 b, u64 c, span<u64> &x, const span<u64> &w, const span<u64> &bias)
    {
        // Method 1: operation fusion
        auto res_tmp = matmul::call(a, b, c, x, w);
        res_tmp = add::call(res_tmp, bias);
        auto res = truncate::call_64_16(res_tmp, FLOAT_PRECISION_64);
        return res;

        // Method 2: do not double the precision of bias, extra one round
        // auto res_tmp = matmul::call(a, b, c, x, w);
        // res_tmp = truncate::call(res_tmp, FLOAT_PRECISION_64);
        // res_tmp = add::call(res_tmp, bias);
        // auto res = down::call_64_16(res_tmp, FLOAT_PRECISION_64, FLOAT_PRECISION_16);
        // return res;
    }

    span<u64> ffn(span<u64> &x, int layer, BertModel &model)
    {
        u64 n_token = x.size() / model.n_embd;

        auto res_tmp = linearDowncast16(n_token, model.n_embd, model.n_interm, x, model.ffn_up_w[layer], model.ffn_up_b[layer]);
        // auto res_tmp = linearDowncast32(n_token, model.n_embd, model.n_interm, x, model.ffn_up_w[layer], model.ffn_up_b[layer]);

        utils::start_timer("nonlinear");
        res_tmp = bert_gelu(res_tmp);
        utils::stop_timer("nonlinear");

        // auto res = up::call_32_64(res_tmp, FLOAT_PRECISION_32, FLOAT_PRECISION_64);
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


    span<u64> mha(span<u64> &x, int layer, BertModel &model, const span<u32> &mask)
    {
        u64 n_token = x.size() / model.n_embd;
        auto c = linear(n_token, model.n_embd, model.n_embd * 3, x, model.c_attn_w[layer], model.c_attn_b[layer]);

        auto q = view(c, n_token, 3, 0);
        auto k = view(c, n_token, 3, 1);
        auto v = view(c, n_token, 3, 2);

        // Batched MHA Optimization:
        u64 head_size = model.n_embd / model.n_heads;
        
        // 1. Compute Attention Scores (Q * K^T)
        // Q_i: [n_token, head_size]
        // K_i^T: [head_size, n_token]
        span<u64> batched_q(model.n_heads * n_token * head_size);
        span<u64> batched_kt(model.n_heads * head_size * n_token);
        span<u64> batched_v(model.n_heads * n_token * head_size);

        // Optimized Q and V preparation
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < model.n_heads; i++)
        {
            for(int r = 0; r < n_token; ++r) 
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
                u64* batched_kt_pointer = batched_kt.data() + i * head_size * n_token + c * n_token;
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
        // Mask shape: [n_token] (0 for valid, -1000 for padding)
        // Broadcast to [n_heads, n_token, n_token]
        // The mask is applied to the key dimension (j), so each query position i attends to key positions j according to the padding mask
        if (mask.data() != nullptr && mask.size() > 0) 
        {
            // Assume mask is 1D [n_token] and masks keys (columns)
            // The mask values should be 0 for valid tokens and a large negative for padding
            // This ensures padding tokens don't contribute to attention

            // #if (DEBUG)
            // {
            //     // DEBUG: Print mask information
            //     std::cout << "[DEBUG MHA] Applying attention mask, size=" << mask.size() << std::endl;
            //     std::cout << "[DEBUG MHA] Mask values (first 10): ";
            //     for (int j = 0; j < std::min(10, (int)mask.size()); ++j) {
            //         std::cout << fix2double(mask[j]) << " ";
            //     }
            //     std::cout << std::endl;

            //     // Count valid vs padding tokens
            //     int valid_count = 0, padding_count = 0;
            //     for (size_t j = 0; j < mask.size(); ++j) {
            //         if (mask[j] == 0) valid_count++;
            //         else padding_count++;
            //     }
            //     std::cout << "[DEBUG MHA] Valid tokens: " << valid_count
            //             << ", Padding tokens: " << padding_count << std::endl;
            //     }
            // #endif

            #pragma omp parallel for collapse(3)
            for(int h=0; h<model.n_heads; ++h) 
            {
                for(int i=0; i<n_token; ++i) 
                {
                    for(int j=0; j<n_token; ++j) 
                    {
                        u64 idx = h * n_token * n_token + i * n_token + j;
                        // Add mask to score. For padding positions (mask[j] < 0), this pushes score to -inf
                        all_scores[idx] = all_scores[idx] + mask[j];
                    }
                }
            }
        } 
        else 
        {
            #if (DEBUG)
            {
                std::cout << "[DEBUG MHA] No attention mask applied (mask.data()="
                    << mask.data() << ", mask.size()=" << mask.size() << ")" << std::endl;
            }
            #endif
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
        for(int r = 0; r < n_token; ++r) 
        {
            for(int i = 0; i < model.n_heads; ++i) 
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

    span<u64> layer(span<u64> &x, int layer, BertModel &model, const span<u32> &mask)
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

    span<u64> bert_classifier(span<u64> &hidden_states, BertModel &model)
    {
        // Extract [CLS] token (first token)
        // hidden_states: [n_token, n_embd]
        // CLS: [1, n_embd]
        span<u64> cls_token(model.n_embd);
        
        // Copy first token
        std::memcpy(cls_token.data(), hidden_states.data(), model.n_embd * sizeof(u64));
        
        // Classifier: Linear [1, n_embd] * [n_embd, num_labels] -> [1, num_labels]
        auto logits = linear(1, model.n_embd, model.num_labels, cls_token, model.classifier_w, model.classifier_b);
        
        return logits;
    }

    span<u64> inference(const span<u64> &x, BertModel &model, const span<u32> &mask)
    {
        span<u64> y = x;
        for (int i = 0; i < model.n_layers; i++)
        {
            // if (party == SERVER) std::cout << "[INFO] Processing Layer " << (i + 1) << "/" << model.n_layers << "..." << std::endl;

            y = layer(y, i, model, mask);
        }

        if (model.accuracy_test)
        {
            // Run Classifier
            // if (party == SERVER) std::cout << "[INFO] Processing Classifier Head..." << std::endl;
            
            auto logits = bert_classifier(y, model);

            return logits;
        }
        else
        {
            return y;
        }
    }

    void share_model_weights(BertModel &model, int owner)
    {
        // if (party == SERVER) std::cout << "[INFO] Sharing model weights..." << std::endl;

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
            // Share Classifier Head
            input::call(model.classifier_w, owner);
            input::call(model.classifier_b, owner);
        }
    }
}