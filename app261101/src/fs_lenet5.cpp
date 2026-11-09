#include "fs_lenet5.hpp"


void fs::lenet5::load_params(const params& p, int owner)
{
    always_assert(owner == sp::SERVER || owner == sp::CLIENT);
    load_(c1_w_, p.c1_w, owner);
    load_(c1_b_, p.c1_b, owner);
    load_(s2_w_, p.s2_w, owner);
    load_(s2_b_, p.s2_b, owner);

    shark::span<u64> connected(C3_CONNECTED_WEIGHT_SIZE_);
    load_(connected, p.c3_w, owner);
    expand_c3_weights_(connected);
    load_(c3_b_, p.c3_b, owner);

    load_(s4_w_, p.s4_w, owner);
    load_(s4_b_, p.s4_b, owner);
    load_(c5_w_, p.c5_w, owner);
    load_(c5_b_, p.c5_b, owner);
    load_(f6_w_, p.f6_w, owner);
    load_(f6_b_, p.f6_b, owner);
    load_(rbf_templates_, p.rbf_templates, owner);
    params_loaded_ = true;
}

shark::span<u64> fs::lenet5::inf(const shark::span<u64>& input) 
const
{
    always_assert(params_loaded_);
    const auto c1 = forward_c1_(input);
    const auto s2 = forward_s2_(c1);
    const auto c3 = forward_c3_(s2);
    const auto s4 = forward_s4_(c3);
    const auto c5 = forward_c5_(s4);
    const auto f6 = forward_f6_(c5);
    const auto distances = forward_rbf_(f6);
    return argmin_(distances);
}


void fs::lenet5::load_(shark::span<u64>& dst, std::span<const u64> src, int owner)
{
    if (sp::party == owner)
    {
        always_assert(src.size() == dst.size());
        for (std::size_t i = 0; i < dst.size(); ++i)
            dst[i] = src[i];
    }
    sp::input::call(dst, owner);
}


static u64 public_value_(u64 value)
{
    return sp::party == sp::DEALER ? 0 : value;
}

shark::span<u64> fs::lenet5::forward_c1_(const shark::span<u64>& input)
const
{
    always_assert(input.size() == C1_INPUT_SIZE_);
    auto z = sp::conv::call
    (
        KERNEL_SIDE_, 
        PADDING_, 
        STRIDE_,
        C1_IN_CHANNELS_, 
        C1_OUT_CHANNELS_, 
        C1_INPUT_SIDE_, 
        C1_INPUT_SIDE_,
        input, c1_w_
    );

    z = sp::truncate::call(z, FLOAT_PRECISION_64);
    auto pre_activation = sp::add::call(z, c1_b_);

    return fs::scaled_tanh::call(pre_activation);
}


shark::span<u64> fs::lenet5::subsample_
(
    std::size_t side, 
    std::size_t channels,
    const shark::span<u64>& input, 
    const shark::span<u64>& weights, 
    const shark::span<u64>& bias
)
{
    auto sum = sp::sumpool::call
    (
        POOL_KERNEL_SIDE_, 
        POOL_PADDING_,
        POOL_STRIDE_, 
        channels, 
        side, 
        side, 
        input
    );
    
    shark::span<u64> scale(sum.size());
    for (std::size_t i = 0; i < scale.size(); ++i)
    {
        scale[i] = weights[i % channels];
    }

    auto z = sp::mul::call(sum, scale);
    z = sp::truncate::call(z, FLOAT_PRECISION_64);
    auto pre_activation = sp::add::call(z, bias);
    
    return fs::scaled_tanh::call(pre_activation);
}

shark::span<u64> fs::lenet5::forward_s2_(const shark::span<u64>& input) 
const
{
    always_assert(input.size() == S2_INPUT_SIZE_);
    return subsample_(S2_INPUT_SIDE_, S2_CHANNELS_, input, s2_w_, s2_b_);
}


void fs::lenet5::expand_c3_weights_(const shark::span<u64>& connected)
{
    std::size_t src = 0;
    std::size_t dst = 0;
    for (const auto connections : C3_CONNECTIONS_)
    {
        for (std::size_t k = 0; k < KERNEL_SIDE_ * KERNEL_SIDE_; ++k)
        {
            for (std::size_t ci = 0; ci < C3_IN_CHANNELS_; ++ci)
            {
                c3_w_[dst++] = (connections & (1u << ci)) ? connected[src++] : 0;
            }
        }
    }

    always_assert(src == connected.size() && dst == c3_w_.size());
}

shark::span<u64> fs::lenet5::forward_c3_(const shark::span<u64>& input) 
const
{
    always_assert(input.size() == C3_INPUT_SIZE_);
    auto z = sp::conv::call
    (
        KERNEL_SIDE_, 
        PADDING_, 
        STRIDE_,
        C3_IN_CHANNELS_, 
        C3_OUT_CHANNELS_, 
        C3_INPUT_SIDE_, 
        C3_INPUT_SIDE_,
        input, c3_w_
    );
    
    z = sp::truncate::call(z, FLOAT_PRECISION_64);
    auto pre_activation = sp::add::call(z, c3_b_);
    
    return fs::scaled_tanh::call(pre_activation);
}

shark::span<u64> fs::lenet5::forward_s4_(const shark::span<u64>& input) 
const
{
    always_assert(input.size() == S4_INPUT_SIZE_);
    return subsample_(S4_INPUT_SIDE_, S4_CHANNELS_, input, s4_w_, s4_b_);
}

shark::span<u64> fs::lenet5::forward_c5_(const shark::span<u64>& input) 
const
{
    always_assert(input.size() == C5_INPUT_SIZE_);
    auto z = sp::conv::call
    (
        KERNEL_SIDE_, 
        PADDING_, 
        STRIDE_,
        C5_IN_CHANNELS_, 
        C5_OUT_CHANNELS_, 
        C5_INPUT_SIDE_, 
        C5_INPUT_SIDE_,
        input, c5_w_
    );

    z = sp::truncate::call(z, FLOAT_PRECISION_64);
    auto pre_activation = sp::add::call(z, c5_b_);
    
    return fs::scaled_tanh::call(pre_activation);
}

shark::span<u64> fs::lenet5::forward_f6_(const shark::span<u64>& input) 
const
{
    always_assert(input.size() == F6_INPUT_SIZE_);
    auto z = sp::matmul::call
    (
        1, 
        F6_INPUT_SIZE_, 
        F6_OUTPUT_SIZE_, 
        input, 
        f6_w_
    );

    z = sp::truncate::call(z, FLOAT_PRECISION_64);
    auto pre_activation = sp::add::call(z, f6_b_);

    return fs::scaled_tanh::call(pre_activation);
}

shark::span<u64> fs::lenet5::forward_rbf_(const shark::span<u64>& input) 
const
{
    always_assert(input.size() == F6_OUTPUT_SIZE_);
    shark::span<u64> delta(RBF_TEMPLATE_SIZE_);
    for (std::size_t i = 0; i < delta.size(); ++i)
    {
        delta[i] = input[i % F6_OUTPUT_SIZE_] - rbf_templates_[i];
    }

    const auto squared = sp::mul::call(delta, delta);
    shark::span<u64> distances(CLASS_COUNT_);
    
    for (std::size_t digit = 0; digit < CLASS_COUNT_; ++digit)
    {
        distances[digit] = 0;
        for (std::size_t i = 0; i < F6_OUTPUT_SIZE_; ++i)
        {
            distances[digit] += squared[digit * F6_OUTPUT_SIZE_ + i];
        }
    }
    
    return distances;
}


shark::span<u64> fs::lenet5::argmin_(const shark::span<u64>& distances)
{
    shark::span<u64> label(1), comparison(1), updates(2);
    shark::span<shark::u8> selectors(2);
    label[0] = 0;
    
    auto best_distance = distances[0];
    const auto one = public_value_(1);

    for (std::size_t digit = 1; digit < CLASS_COUNT_; ++digit)
    {
        comparison[0] = best_distance - distances[digit] - one;
        const auto take = sp::drelu::call(comparison);
        selectors[0] = selectors[1] = take[0];
        updates[0] = distances[digit] - best_distance;
        updates[1] = public_value_(digit) - label[0];
        const auto selected = sp::select::call(selectors, updates);
        best_distance += selected[0];
        label[0] += selected[1];
    }
    
    return label;
}