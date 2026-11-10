#include "fs_bert_cls.hpp"

#include <shark/protocols/finalize.hpp>
#include <shark/protocols/init.hpp>
#include <shark/protocols/input.hpp>
#include <shark/protocols/output.hpp>

#include <cnpy.h>

#include <algorithm>
#include <cstddef>
#include <exception>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

namespace
{
    namespace protocol = shark::protocols;
    using shark::u32;
    using shark::u64;

    constexpr std::size_t TOKEN_COUNT = 128;
    constexpr std::size_t HIDDEN_SIZE = 768;
    constexpr std::size_t FFN_SIZE = 3072;
    constexpr std::size_t LABEL_COUNT = 2;

    template <typename T>
    std::span<const T> read_array(const cnpy::npz_t& archive, const std::string& name,
                                  std::initializer_list<std::size_t> shape)
    {
        const auto entry = archive.find(name);
        if (entry == archive.end())
            throw std::runtime_error("missing NPZ array: " + name);

        const auto& array = entry->second;
        if (array.word_size != sizeof(T) || array.fortran_order ||
            !std::equal(array.shape.begin(), array.shape.end(), shape.begin(), shape.end()))
        {
            throw std::runtime_error("unexpected NPZ array layout: " + name);
        }

        return {array.data<T>(), array.num_vals};
    }

    void load_weights(fs::bert_cls& model, const char* path)
    {
        const auto archive = cnpy::npz_load(path);
        fs::bert_cls::params params;
        for (std::size_t i = 0; i < params.layers.size(); ++i)
        {
            const auto prefix = "l" + std::to_string(i) + "_";
            params.layers[i] = {
                .c_attn_w = read_array<u64>(archive, prefix + "c_attn_w", {HIDDEN_SIZE, 3 * HIDDEN_SIZE}),
                .c_attn_b = read_array<u64>(archive, prefix + "c_attn_b", {3 * HIDDEN_SIZE}),
                .c_proj_w = read_array<u64>(archive, prefix + "c_proj_w", {HIDDEN_SIZE, HIDDEN_SIZE}),
                .c_proj_b = read_array<u64>(archive, prefix + "c_proj_b", {HIDDEN_SIZE}),
                .ffn_up_w = read_array<u64>(archive, prefix + "ffn_up_w", {HIDDEN_SIZE, FFN_SIZE}),
                .ffn_up_b = read_array<u64>(archive, prefix + "ffn_up_b", {FFN_SIZE}),
                .ffn_down_w = read_array<u64>(archive, prefix + "ffn_down_w", {FFN_SIZE, HIDDEN_SIZE}),
                .ffn_down_b = read_array<u64>(archive, prefix + "ffn_down_b", {HIDDEN_SIZE}),
                .ln1_w = read_array<u64>(archive, prefix + "ln1_w", {HIDDEN_SIZE}),
                .ln1_b = read_array<u64>(archive, prefix + "ln1_b", {HIDDEN_SIZE}),
                .ln2_w = read_array<u64>(archive, prefix + "ln2_w", {HIDDEN_SIZE}),
                .ln2_b = read_array<u64>(archive, prefix + "ln2_b", {HIDDEN_SIZE}),
            };
        }
        params.classifier_w = read_array<u64>(archive, "classifier_w", {HIDDEN_SIZE, LABEL_COUNT});
        params.classifier_b = read_array<u64>(archive, "classifier_b", {LABEL_COUNT});
        model.load_params(params, protocol::SERVER);
    }
}

int main(int argc, char** argv)
{
    const auto usage = [&]
    {
        std::cerr << "Usage:\n"
                  << "  " << argv[0] << " 2\n"
                  << "  " << argv[0] << " 0 [weights.npz]\n"
                  << "  " << argv[0] << " 1 <sample.npz>\n";
        return 1;
    };

    if (argc < 2)
        return usage();
    const std::string_view role(argv[1]);
    if (role.size() != 1 || role[0] < '0' || role[0] > '2')
        return usage();
    const int pid = role[0] - '0';
    if ((pid == protocol::DEALER && argc != 2) ||
        (pid == protocol::SERVER && argc > 3) ||
        (pid == protocol::CLIENT && argc != 3))
        return usage();

    try
    {
        shark::span<u64> input(TOKEN_COUNT * HIDDEN_SIZE);
        shark::span<u32> mask(TOKEN_COUNT);
        if (pid == protocol::CLIENT)
        {
            const auto sample = cnpy::npz_load(argv[2]);
            const auto embeddings = read_array<u64>(sample, "input", {TOKEN_COUNT, HIDDEN_SIZE});
            const auto attention_mask = read_array<u32>(sample, "attention_mask", {TOKEN_COUNT});
            std::copy(embeddings.begin(), embeddings.end(), input.begin());
            std::copy(attention_mask.begin(), attention_mask.end(), mask.begin());
        }

        if (pid == protocol::DEALER)
        {
            std::random_device random;
            protocol::init::gen(std::uniform_int_distribution<u64>{}(random));
        }
        else
        {
            protocol::init::eval(pid, "127.0.0.1", 42069);
        }

        fs::bert_cls model;
        if (pid == protocol::SERVER)
            load_weights(model, argc == 3 ? argv[2] : "raw_data/bert_weights_fixed.npz");
        else
            model.load_params({}, protocol::SERVER);

        protocol::input::call(input, protocol::CLIENT);
        protocol::input::call(mask, protocol::CLIENT);
        auto logits = model.inf(input, mask);
        protocol::output::call(logits);
        protocol::finalize::call();

        if (pid == protocol::CLIENT)
        {
            const double logit0 = fs::bert_cls::fix2double(logits[0]);
            const double logit1 = fs::bert_cls::fix2double(logits[1]);
            std::cout << std::setprecision(10) << "logits=[" << logit0 << ", " << logit1
                      << "] prediction=" << (logit1 > logit0 ? 1 : 0) << '\n';
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "BERT: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
