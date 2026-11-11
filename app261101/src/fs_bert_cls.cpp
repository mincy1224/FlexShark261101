#include "fs_bert_cls.hpp"
#include "fs_bert_fv.hpp"

#include <algorithm>

namespace sp = shark::protocols;

struct fs::bert_cls::model : fs::BertModel
{
    using BertModel::BertModel;
};

fs::bert_cls::bert_cls()
    : model_(std::make_unique<model>(NUM_LAYERS_, NUM_HEADS_, HIDDEN_SIZE_, FFN_SIZE_, NUM_LABELS_, true))
{
}

fs::bert_cls::~bert_cls() = default;

double fs::bert_cls::fix2double(shark::u64 x)
{
    return fs::fix2double(x);
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

    fs::share_model_weights(*model_, owner);
    params_loaded_ = true;
}

shark::span<shark::u64> fs::bert_cls::inf(
    const shark::span<shark::u64>& input,
    const shark::span<shark::u32>& mask) const
{
    always_assert(params_loaded_);
    always_assert(input.size() > 0 && input.size() % HIDDEN_SIZE_ == 0);
    always_assert(mask.size() == 0 || mask.size() == input.size() / HIDDEN_SIZE_);
    return fs::inference(input, *model_, mask);
}
