#include "fs_scaled_tanh.hpp"

namespace p = shark::protocols;

static const std::vector<shark::u64>& fs::scaled_tanh::tanh_table_()
{
    static const auto table = []
    {
        std::vector<shark::u64> values(std::size_t{1} << TANH_TABLE_BITS_);

        constexpr double step = 1.0 / (1 << TANH_INPUT_FRAC_);
        constexpr auto scale = shark::u64{1} << FLOAT_PRECISION_64;

        for (std::size_t i = 0; i < values.size(); ++i)
        {
            const double x = std::fmin(i * step - TANH_BOUND_, TANH_BOUND_);
            const double y = 1.7159 * std::tanh((2.0 / 3.0) * x);
            values[i] = static_cast<shark::u64>(std::llround(y * scale));
        }
        return values;
    }();

    return table;
}

shark::span<shark::u64> fs::scaled_tanh::call(const shark::span<shark::u64> &input)
{

    constexpr shark::u64 scale = shark::u64{1} << FLOAT_PRECISION_64;

    const shark::u64 bound = p::party == p::DEALER ? 0 : TANH_BOUND_ * scale;
    const std::size_t size = input.size();

    shark::span<shark::u64> shifted(2 * size);
    for (std::size_t i = 0; i < size; ++i)
    {
        shifted[i] = input[i] + bound;
        shifted[size + i] = input[i] - bound;
    }
    const auto positive = p::relu::call(shifted);

    shark::span<shark::u64> offset(size);
    for (std::size_t i = 0; i < size; ++i)
    {
        offset[i] = positive[i] - positive[size + i];
    }

    const auto index = p::truncate::call
    (
        offset, 
        FLOAT_PRECISION_64 - TANH_INPUT_FRAC_
    );

    return p::lut::call(index, tanh_table_(), TANH_TABLE_BITS_);
}