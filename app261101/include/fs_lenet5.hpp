#ifndef FLEX_SHARK_LENET_5_HPP
#define FLEX_SHARK_LENET_5_HPP

#include "shark/types/span.hpp"
#include "shark/types/u128.hpp"
#include "shark/types/u16.hpp"
#include "shark/types/u64.hpp"
#include <array>
#include <cstddef>
#include <span>
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

    class lenet5
    {
    public:
        explicit lenet5
        (
            // 模型尺寸 
        ){}
        
        void inf(const shark::span<u64>& input);

    private: 
        static constexpr size_t KERNEL_SIDE_ = 5;
        static constexpr size_t STRIDE_ = 1;
        static constexpr size_t PADDING_ = 0;
        static constexpr size_t INPUT_SIDE_ = 32;

        // C1
        static constexpr size_t C1_INPUT_SIDE_ = INPUT_SIDE_;
        static constexpr size_t C1_IN_CHANNELS_ = 1;
        static constexpr size_t C1_OUT_CHANNELS_ = 6;

        static constexpr size_t C1_INPUT_SIZE_ = C1_INPUT_SIDE_ * C1_INPUT_SIDE_ * C1_IN_CHANNELS_;
        static constexpr size_t C1_WEIGHT_SIZE_ = C1_OUT_CHANNELS_ * KERNEL_SIDE_ * KERNEL_SIDE_ * C1_IN_CHANNELS_;

        shark::span<shark::u64> c1_d_{C1_INPUT_SIZE_}; // [h, w, c_in]
        shark::span<shark::u64> c1_w_{C1_WEIGHT_SIZE_};  // [c_out, k, k, c_in]

        [[nodiscard]]
        shark::span<shark::u64> forward_c1_(const shark::span<shark::u64>& input) 
        {
            always_assert(input.size() == C1_INPUT_SIZE_);
            // conv::mod::call
            // (
            //     KERNEL_SIDE_,
            //     PADDING_,
            //     STRIDE_,
            //     C1_IN_CHANNELS_,
            //     C1_OUT_CHANNELS_,
            //     INPUT_SIDE_,
            //     INPUT_SIDE_,
            //     input,
            //     c1_w_,
            //     c1_d_
            // );
            auto z = conv::call
            (
                KERNEL_SIDE_,
                PADDING_,
                STRIDE_,
                C1_IN_CHANNELS_,
                C1_OUT_CHANNELS_,
                INPUT_SIDE_,
                INPUT_SIDE_,
                input,
                c1_w_
            );

            // need truncate ?
            z = truncate::call(z, FLOAT_PRECISION_64);

            return add::call(z, c1_d_);

        }

    };

};

#endif // !FLEX_SHARK_LENET_5_HPP