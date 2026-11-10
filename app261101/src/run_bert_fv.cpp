#include "fs_bert_fv.hpp"

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

    void load_(shark::span<u64>& dst, std::span<const u64> src)
    {
        if (src.size() != dst.size())
            throw std::runtime_error("weight size mismatch");
        std::copy(src.begin(), src.end(), dst.begin());
    }

    void load_weights(fs::BertModel& model, const char* path)
    {
        const auto archive = cnpy::npz_load(path);
        for (std::size_t i = 0; i < model.n_layers; ++i)
        {
            const auto prefix = "l" + std::to_string(i) + "_";
            load_(model.c_attn_w[i], read_array<u64>(archive, prefix + "c_attn_w", {HIDDEN_SIZE, 3 * HIDDEN_SIZE}));
            load_(model.c_attn_b[i], read_array<u64>(archive, prefix + "c_attn_b", {3 * HIDDEN_SIZE}));
            load_(model.c_proj_w[i], read_array<u64>(archive, prefix + "c_proj_w", {HIDDEN_SIZE, HIDDEN_SIZE}));
            load_(model.c_proj_b[i], read_array<u64>(archive, prefix + "c_proj_b", {HIDDEN_SIZE}));
            load_(model.ffn_up_w[i], read_array<u64>(archive, prefix + "ffn_up_w", {HIDDEN_SIZE, FFN_SIZE}));
            load_(model.ffn_up_b[i], read_array<u64>(archive, prefix + "ffn_up_b", {FFN_SIZE}));
            load_(model.ffn_down_w[i], read_array<u64>(archive, prefix + "ffn_down_w", {FFN_SIZE, HIDDEN_SIZE}));
            load_(model.ffn_down_b[i], read_array<u64>(archive, prefix + "ffn_down_b", {HIDDEN_SIZE}));
            load_(model.ln1_w[i], read_array<u64>(archive, prefix + "ln1_w", {HIDDEN_SIZE}));
            load_(model.ln1_b[i], read_array<u64>(archive, prefix + "ln1_b", {HIDDEN_SIZE}));
            load_(model.ln2_w[i], read_array<u64>(archive, prefix + "ln2_w", {HIDDEN_SIZE}));
            load_(model.ln2_b[i], read_array<u64>(archive, prefix + "ln2_b", {HIDDEN_SIZE}));
        }
        load_(model.classifier_w, read_array<u64>(archive, "classifier_w", {HIDDEN_SIZE, LABEL_COUNT}));
        load_(model.classifier_b, read_array<u64>(archive, "classifier_b", {LABEL_COUNT}));
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

    // --stream selects oneShot=false; every other argument is positional.
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
            const auto sample = cnpy::npz_load(positional[1]);
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

        fs::BertModel model(12, 12, HIDDEN_SIZE, FFN_SIZE, LABEL_COUNT, true);
        if (pid == protocol::SERVER)
            load_weights(model, count == 2 ? positional[1] : "raw_data/bert_weights_fixed.npz");
        fs::share_model_weights(model, protocol::SERVER);

        protocol::input::call(input, protocol::CLIENT);
        protocol::input::call(mask, protocol::CLIENT);
        auto logits = fs::inference(input, model, mask);
        protocol::output::call(logits);
        protocol::finalize::call();

        if (pid == protocol::CLIENT)
        {
            const double logit0 = fs::fix2double(logits[0]);
            const double logit1 = fs::fix2double(logits[1]);
            std::cout << std::setprecision(10) << "logits=[" << logit0 << ", " << logit1
                      << "] prediction=" << (logit1 > logit0 ? 1 : 0) << '\n';
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "BERT-FV: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
