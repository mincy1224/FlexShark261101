#ifndef FLEX_SHARK_LENET_5_HPP
#define FLEX_SHARK_LENET_5_HPP

#include <array>
#include <cstddef>
#include <span>

#include <shark/types/span.hpp>
#include <shark/utils/assert.hpp>
#include <shark/protocols/add.hpp>
#include <shark/protocols/common.hpp>
#include <shark/protocols/conv.hpp>
#include <shark/protocols/drelu.hpp>
#include <shark/protocols/input.hpp>
#include <shark/protocols/matmul.hpp>
#include <shark/protocols/mul.hpp>
#include <shark/protocols/select.hpp>
#include <shark/protocols/sumpool.hpp>
#include <shark/protocols/truncate.hpp>
#include <shark/utils/globals.hpp>

#include "fs_scaled_tanh.hpp"

namespace sp = shark::protocols;
using shark::u64;

namespace fs
{
    class lenet5
    {
    public:
        struct params
        {
            std::span<const u64> c1_w{}, c1_b{};
            std::span<const u64> s2_w{}, s2_b{};
            std::span<const u64> c3_w{}, c3_b{}; 
            std::span<const u64> s4_w{}, s4_b{};
            std::span<const u64> c5_w{}, c5_b{};
            std::span<const u64> f6_w{}, f6_b{};
            std::span<const u64> rbf_templates{};
        };

        void load_params(const params& p, int owner = sp::SERVER);

        [[nodiscard]] 
        shark::span<u64> inf(const shark::span<u64>& input) 
        const;

    private:
        static constexpr std::size_t KERNEL_SIDE_ = 5;
        static constexpr std::size_t STRIDE_ = 1;
        static constexpr std::size_t PADDING_ = 0;
        static constexpr std::size_t INPUT_SIDE_ = 32;
        static constexpr std::size_t POOL_KERNEL_SIDE_ = 2;
        static constexpr std::size_t POOL_STRIDE_ = 2;
        static constexpr std::size_t POOL_PADDING_ = 0;

        static void load_(shark::span<u64>& dst, std::span<const u64> src, int owner);


        // === C1 ===
        static constexpr std::size_t C1_INPUT_SIDE_ = INPUT_SIDE_;
        static constexpr std::size_t C1_IN_CHANNELS_ = 1;
        static constexpr std::size_t C1_OUT_CHANNELS_ = 6;
        static constexpr std::size_t C1_INPUT_SIZE_ = C1_INPUT_SIDE_ * C1_INPUT_SIDE_ * C1_IN_CHANNELS_;
        static constexpr std::size_t C1_WEIGHT_SIZE_ = C1_OUT_CHANNELS_ * KERNEL_SIDE_ * KERNEL_SIDE_ * C1_IN_CHANNELS_;
        static constexpr std::size_t C1_OUTPUT_SIDE_ = (C1_INPUT_SIDE_ + 2 * PADDING_ - KERNEL_SIDE_) / STRIDE_ + 1;
        shark::span<u64> c1_w_{C1_WEIGHT_SIZE_}; // [6, 5, 5, 1]
        shark::span<u64> c1_b_{C1_OUT_CHANNELS_};

        [[nodiscard]] 
        shark::span<u64> forward_c1_(const shark::span<u64>& input)
        const;

        [[nodiscard]] 
        static shark::span<u64> subsample_
        (
            std::size_t side, 
            std::size_t channels,
            const shark::span<u64>& input, 
            const shark::span<u64>& weights, 
            const shark::span<u64>& bias
        );

        // === S2 ===
        static constexpr std::size_t S2_INPUT_SIDE_     = C1_OUTPUT_SIDE_;
        static constexpr std::size_t S2_CHANNELS_       = C1_OUT_CHANNELS_;
        static constexpr std::size_t S2_INPUT_SIZE_     = S2_INPUT_SIDE_ * S2_INPUT_SIDE_ * S2_CHANNELS_;
        static constexpr std::size_t S2_OUTPUT_SIDE_    = (S2_INPUT_SIDE_ + 2 * POOL_PADDING_ - POOL_KERNEL_SIDE_) / POOL_STRIDE_ + 1;
        static constexpr std::size_t S2_OUTPUT_SIZE_    = S2_OUTPUT_SIDE_ * S2_OUTPUT_SIDE_ * S2_CHANNELS_;
        shark::span<u64> s2_w_{S2_CHANNELS_};
        shark::span<u64> s2_b_{S2_CHANNELS_};

        [[nodiscard]] 
        shark::span<u64> forward_s2_(const shark::span<u64>& input) 
        const;


        // === C3 ===
        static constexpr std::size_t C3_INPUT_SIDE_     = S2_OUTPUT_SIDE_;
        static constexpr std::size_t C3_IN_CHANNELS_    = S2_CHANNELS_;
        static constexpr std::size_t C3_OUT_CHANNELS_   = 16;
        static constexpr std::size_t C3_INPUT_SIZE_     = S2_OUTPUT_SIZE_;
        static constexpr std::size_t C3_WEIGHT_SIZE_    = C3_OUT_CHANNELS_ * KERNEL_SIDE_ * KERNEL_SIDE_ * C3_IN_CHANNELS_;
        static constexpr std::size_t C3_CONNECTED_WEIGHT_SIZE_ = 1500;
        static constexpr std::size_t C3_OUTPUT_SIDE_    = (C3_INPUT_SIDE_ + 2 * PADDING_ - KERNEL_SIDE_) / STRIDE_ + 1;
        static constexpr std::size_t C3_OUTPUT_SIZE_    = C3_OUTPUT_SIDE_ * C3_OUTPUT_SIDE_ * C3_OUT_CHANNELS_;

        // C3 的通道连接表，这里参考的是1998的原文
        static constexpr std::array<unsigned, C3_OUT_CHANNELS_> C3_CONNECTIONS_
        {
            0x07, 0x0e, 0x1c, 0x38, 0x31, 0x23,
            0x0f, 0x1e, 0x3c, 0x39, 0x33, 0x27,
            0x1b, 0x36, 0x2d, 0x3f
        };
        
        shark::span<u64> c3_w_{C3_WEIGHT_SIZE_}; // [16, 5, 5, 6]
        shark::span<u64> c3_b_{C3_OUT_CHANNELS_};

        void expand_c3_weights_(const shark::span<u64>& connected);

        [[nodiscard]] 
        shark::span<u64> forward_c3_(const shark::span<u64>& input) 
        const;


        // === S4 ===
        static constexpr std::size_t S4_INPUT_SIDE_     = C3_OUTPUT_SIDE_;
        static constexpr std::size_t S4_CHANNELS_       = C3_OUT_CHANNELS_;
        static constexpr std::size_t S4_INPUT_SIZE_     = C3_OUTPUT_SIZE_;
        static constexpr std::size_t S4_OUTPUT_SIDE_    = (S4_INPUT_SIDE_ + 2 * POOL_PADDING_ - POOL_KERNEL_SIDE_) / POOL_STRIDE_ + 1;
        static constexpr std::size_t S4_OUTPUT_SIZE_    = S4_OUTPUT_SIDE_ * S4_OUTPUT_SIDE_ * S4_CHANNELS_;
        shark::span<u64> s4_w_{S4_CHANNELS_};
        shark::span<u64> s4_b_{S4_CHANNELS_};

        [[nodiscard]] 
        shark::span<u64> forward_s4_(const shark::span<u64>& input) 
        const;
        

        // === C5 ===
        static constexpr std::size_t C5_INPUT_SIDE_     = S4_OUTPUT_SIDE_;
        static constexpr std::size_t C5_IN_CHANNELS_    = S4_CHANNELS_;
        static constexpr std::size_t C5_OUT_CHANNELS_   = 120;
        static constexpr std::size_t C5_INPUT_SIZE_     = S4_OUTPUT_SIZE_;
        static constexpr std::size_t C5_WEIGHT_SIZE_    = C5_OUT_CHANNELS_ * KERNEL_SIDE_ * KERNEL_SIDE_ * C5_IN_CHANNELS_;
        static constexpr std::size_t C5_OUTPUT_SIDE_    = (C5_INPUT_SIDE_ + 2 * PADDING_ - KERNEL_SIDE_) / STRIDE_ + 1;
        static constexpr std::size_t C5_OUTPUT_SIZE_    = C5_OUTPUT_SIDE_ * C5_OUTPUT_SIDE_ * C5_OUT_CHANNELS_;
        shark::span<u64> c5_w_{C5_WEIGHT_SIZE_}; // [120, 5, 5, 16]
        shark::span<u64> c5_b_{C5_OUT_CHANNELS_};

        [[nodiscard]] 
        shark::span<u64> forward_c5_(const shark::span<u64>& input) 
        const;


        // === F6 ===
        static constexpr std::size_t F6_INPUT_SIZE_     = C5_OUTPUT_SIZE_;
        static constexpr std::size_t F6_OUTPUT_SIZE_    = 84;
        static constexpr std::size_t F6_WEIGHT_SIZE_    = F6_INPUT_SIZE_ * F6_OUTPUT_SIZE_;
        shark::span<u64> f6_w_{F6_WEIGHT_SIZE_};    // [120, 84]
        shark::span<u64> f6_b_{F6_OUTPUT_SIZE_};

        [[nodiscard]] 
        shark::span<u64> forward_f6_(const shark::span<u64>& input) 
        const;

        static constexpr std::size_t CLASS_COUNT_       = 10;
        static constexpr std::size_t RBF_TEMPLATE_SIZE_ = CLASS_COUNT_ * F6_OUTPUT_SIZE_;
        shark::span<u64> rbf_templates_{RBF_TEMPLATE_SIZE_};

        [[nodiscard]] 
        shark::span<u64> forward_rbf_(const shark::span<u64>& input) 
        const;

        [[nodiscard]] 
        static shark::span<u64> argmin_(const shark::span<u64>& distances);

        bool params_loaded_ = false;
    };
}

#endif // FLEX_SHARK_LENET_5_HPP
