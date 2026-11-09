#ifndef FLEX_SHARK_SCALED_TANH_HPP
#define FLEX_SHARK_SCALED_TANH_HPP
#include <cmath>
#include <cstddef>
#include <vector>

#include <shark/protocols/common.hpp>
#include <shark/protocols/lut.hpp>
#include <shark/protocols/relu.hpp>
#include <shark/protocols/truncate.hpp>
#include <shark/utils/globals.hpp>


namespace fs::scaled_tanh
{
    inline constexpr int TANH_INPUT_FRAC_ = 8;
    inline constexpr int TANH_TABLE_BITS_ = 12;
    inline constexpr int TANH_BOUND_ = 6;

    static const std::vector<shark::u64>& tanh_table_();

    [[nodiscard]] 
    shark::span<shark::u64> call(const shark::span<shark::u64>& input);
}

#endif // !FLEX_SHARK_SCALED_TANH_HPP