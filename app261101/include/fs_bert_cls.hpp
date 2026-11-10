#ifndef FLEX_SHARK_BERT_CLS_HPP
#define FLEX_SHARK_BERT_CLS_HPP

#include <array>
#include <cstddef>
#include <memory>
#include <span>

#include <shark/protocols/common.hpp>

namespace fs
{
    class bert_cls
    {
    public:
        static constexpr std::size_t NUM_LAYERS_ = 12;
        static constexpr std::size_t NUM_HEADS_ = 12;
        static constexpr std::size_t HIDDEN_SIZE_ = 768;
        static constexpr std::size_t FFN_SIZE_ = 3072;
        static constexpr std::size_t NUM_LABELS_ = 2;

        struct layer_params
        {
            std::span<const shark::u64> c_attn_w{}, c_attn_b{}; // [768, 2304]
            std::span<const shark::u64> c_proj_w{}, c_proj_b{};
            std::span<const shark::u64> ffn_up_w{}, ffn_up_b{}; // ffn_up_b
            std::span<const shark::u64> ffn_down_w{}, ffn_down_b{};
            std::span<const shark::u64> ln1_w{}, ln1_b{};
            std::span<const shark::u64> ln2_w{}, ln2_b{};
        };

        struct params
        {
            std::array<layer_params, NUM_LAYERS_> layers{};
            std::span<const shark::u64> classifier_w{}, classifier_b{};
        };

        struct model;

        bert_cls();
        ~bert_cls();
        bert_cls(const bert_cls&) = delete;
        bert_cls& operator=(const bert_cls&) = delete;

        void load_params(const params& p, int owner = shark::protocols::SERVER);

        [[nodiscard]]
        shark::span<shark::u64> inf
        (
            const shark::span<shark::u64>& input,
            const shark::span<shark::u32>& mask = {}
        ) const;

        static double fix2double(shark::u64 x);

    private:
        static void load_(shark::span<shark::u64>& dst, std::span<const shark::u64> src);

        std::unique_ptr<model> model_;
        bool params_loaded_ = false;
    };
}

#endif // !FLEX_SHARK_BERT_CLS_HPP
