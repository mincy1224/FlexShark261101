#ifndef FLEX_SHARK_BERT_FV_HPP
#define FLEX_SHARK_BERT_FV_HPP

#include <cstring>
#include <shark/protocols/common.hpp>
#include <shark/protocols/init.hpp>
#include <shark/protocols/finalize.hpp>
#include <shark/protocols/input.hpp>
#include <shark/protocols/output.hpp>
#include <shark/protocols/relu.hpp>
#include <shark/protocols/matmul.hpp>
#include <shark/protocols/matmulpar.hpp>
#include <shark/protocols/conv.hpp>
#include <shark/protocols/add.hpp>
#include <shark/protocols/mul.hpp>
#include <shark/protocols/inv.hpp>
#include <shark/protocols/truncate.hpp>
#include <shark/protocols/softmax.hpp>
#include <shark/protocols/gelu.hpp>
#include <shark/protocols/rsqrt.hpp>
#include <shark/protocols/up.hpp>
#include <shark/protocols/down.hpp>
#include <shark/protocols/layernorm.hpp>

#include <shark/utils/globals.hpp>
#include <shark/utils/timer.hpp>
#include <shark/utils/assert.hpp>

using namespace shark;
using namespace shark::protocols;

namespace fs
{        
    struct BertModel
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

        BertModel(u64 layers=12, u64 heads=12, u64 embd=768, u64 interm=3072, u64 labels=2, bool acc_test=false) 
            : n_layers(layers), n_heads(heads), n_embd(embd), n_interm(interm), num_labels(labels), accuracy_test(acc_test),
            c_attn_w(n_layers), c_attn_b(n_layers), c_proj_w(n_layers), c_proj_b(n_layers), 
            ffn_up_w(n_layers), ffn_up_b(n_layers), ffn_down_w(n_layers), ffn_down_b(n_layers),
            ln1_w(n_layers), ln1_b(n_layers), ln2_w(n_layers), ln2_b(n_layers)
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
                // Init classifier head
                classifier_w = span<u64>(n_embd * num_labels);    
                classifier_b = span<u64>(num_labels);
            }
        }
    };

    u64 double2fix(double x);
    double fix2double(u64 x);
    double fix2double(u32 x);

    template <typename T>
    inline span<T> bert_gelu(span<T> &x)
    {
        return gelu::call(x);
    }

    span<u64> bert_softmax(u64 a, u64 b, span<u32> &x);

    span<u64> linear(u64 a, u64 b, u64 c, span<u64> &x, const span<u64> &w, const span<u64> &bias);

    span<u32> linearDowncast32(u64 a, u64 b, u64 c, span<u64> &x, const span<u64> &w, const span<u64> &bias);

    span<u16> linearDowncast16(u64 a, u64 b, u64 c, span<u64> &x, const span<u64> &w, const span<u64> &bias);

    span<u64> ffn(span<u64> &x, int layer, BertModel &model);

    span<u64> bert_layernorm(span<u64> &x, u64 n_token, u64 n_embd, const span<u64> &gamma, const span<u64> &beta);

    template <typename T>
    inline span<T> view(span<T> &x, int n_token, int n_peices, int idx)
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

    template <typename T>
    inline span<T> transpose(u64 a, u64 b, span<T> &x)
    {
        always_assert(x.size() == (a * b));
        span<T> y(x.size());
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < a; i++)
        {
            for (int j = 0; j < b; j++)
            {
                y[j * a + i] = x[i * b + j];
            }
        }

        return y;
    }

    template <typename T>
    inline void concat(span<T> &x, span<T> &y, int n_token, int idx)
    {
        int x_n_embd = x.size() / n_token;
        int y_n_embd = y.size() / n_token;
        int n_peices = x_n_embd / y_n_embd;

        #pragma omp parallel for collapse(2)
        for (int i = 0; i < n_token; i++)
        {
            for (int j = 0; j < y_n_embd; j++)
            {
                x[i * x_n_embd + j + idx * y_n_embd] = y[i * y_n_embd + j];
            }
        }
    }

    span<u64> mha(span<u64> &x, int layer, BertModel &model, const span<u32> &mask = span<u32>());

    span<u64> layer(span<u64> &x, int layer, BertModel &model, const span<u32> &mask = span<u32>());

    span<u64> bert_classifier(span<u64> &hidden_states, BertModel &model);

    span<u64> inference(const span<u64> &x, BertModel &model, const span<u32> &mask = span<u32>());

    void share_model_weights(BertModel &model, int owner = SERVER);
}

#endif // !FLEX_SHARK_BERT_FV_HPP