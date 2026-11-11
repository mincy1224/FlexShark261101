#include "fs_bert_cls.hpp"
#include "bert_fixed_npz.hpp"

#include <shark/protocols/finalize.hpp>
#include <shark/protocols/init.hpp>
#include <shark/protocols/input.hpp>
#include <shark/protocols/output.hpp>

#include <algorithm>
#include <exception>
#include <iomanip>
#include <iostream>
#include <random>
#include <span>
#include <string>
#include <string_view>

namespace
{
    namespace protocol = shark::protocols;
    using shark::u32;
    using shark::u64;
    using namespace fs::bert_fixed_npz;

    void load_weights(fs::bert_cls& model, const char* path)
    {
        const auto archive = load_archive(path);
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
                  << "  " << argv[0] << " [--stream] 2\n"
                  << "  " << argv[0] << " [--stream] 0 [weights.npz]\n"
                  << "  " << argv[0] << " [--stream] 1 <sample.npz>\n"
                  << "\n"
                  << "  --stream   read the dealer key file on demand instead of\n"
                  << "             loading it into memory; needed when the key file\n"
                  << "             is larger than the available RAM\n";
        return 1;
    };

    bool one_shot = true;
    const char* positional[3] = {nullptr, nullptr, nullptr};
    int count = 0;
    for (int i = 1; i < argc; ++i)
    {
        const std::string_view arg(argv[i]);
        if (arg == "--stream" || arg == "--oneshot=0" || arg == "--oneshot=false")
        {
            one_shot = false;
            continue;
        }
        if (arg == "--oneshot=1" || arg == "--oneshot=true")
        {
            one_shot = true;
            continue;
        }
        if (count == 3)
            return usage();
        positional[count++] = argv[i];
    }
    if (count == 0)
        return usage();
    const std::string_view role(positional[0]);
    if (role.size() != 1 || role[0] < '0' || role[0] > '2')
        return usage();
    const int pid = role[0] - '0';
    if ((pid == protocol::DEALER && count != 1) ||
        (pid == protocol::SERVER && count > 2) ||
        (pid == protocol::CLIENT && count != 2))
        return usage();

    try
    {
        shark::span<u64> input(TOKEN_COUNT * HIDDEN_SIZE);
        shark::span<u32> mask(TOKEN_COUNT);
        if (pid == protocol::CLIENT)
        {
            const auto sample = load_archive(positional[1]);
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
            protocol::init::eval(pid, "127.0.0.1", 42069, one_shot);
        }

        fs::bert_cls model;
        if (pid == protocol::SERVER)
            load_weights(model, count == 2 ? positional[1] : DEFAULT_WEIGHTS);
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
