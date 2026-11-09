#include "fs_lenet5.hpp"

#include <shark/protocols/finalize.hpp>
#include <shark/protocols/init.hpp>
#include <shark/protocols/input.hpp>
#include <shark/protocols/output.hpp>

#include <cnpy.h>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <exception>
#include <initializer_list>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace
{
    namespace protocol = shark::protocols;
    using shark::u64;
    using shark::u8;

    template <typename T>
    std::vector<T> read_array(const cnpy::npz_t& archive, const char* name,
                              std::initializer_list<std::size_t> shape)
    {
        const auto entry = archive.find(name);
        if (entry == archive.end())
            throw std::runtime_error(std::string("missing NPZ array: ") + name);

        const auto& array = entry->second;
        if (array.word_size != sizeof(T) || array.fortran_order ||
            !std::equal(array.shape.begin(), array.shape.end(), shape.begin(), shape.end()))
        {
            throw std::runtime_error(std::string("unexpected NPZ array layout: ") + name);
        }

        std::vector<T> values(array.num_vals);
        std::memcpy(values.data(), array.data<char>(), array.num_bytes());
        return values;
    }

    void load_weights(fs::lenet5& model, const char* path)
    {
        const auto archive = cnpy::npz_load(path);
        const auto c1_w = read_array<u64>(archive, "c1_w", {6, 5, 5, 1});
        const auto c1_b = read_array<u64>(archive, "c1_b", {6});
        const auto s2_w = read_array<u64>(archive, "s2_w", {6});
        const auto s2_b = read_array<u64>(archive, "s2_b", {6});
        const auto c3_w = read_array<u64>(archive, "c3_w", {1500});
        const auto c3_b = read_array<u64>(archive, "c3_b", {16});
        const auto s4_w = read_array<u64>(archive, "s4_w", {16});
        const auto s4_b = read_array<u64>(archive, "s4_b", {16});
        const auto c5_w = read_array<u64>(archive, "c5_w", {120, 5, 5, 16});
        const auto c5_b = read_array<u64>(archive, "c5_b", {120});
        const auto f6_w = read_array<u64>(archive, "f6_w", {120, 84});
        const auto f6_b = read_array<u64>(archive, "f6_b", {84});
        const auto rbf_templates = read_array<u64>(archive, "rbf_templates", {10, 84});

        model.load_params({
            .c1_w = c1_w, .c1_b = c1_b,
            .s2_w = s2_w, .s2_b = s2_b,
            .c3_w = c3_w, .c3_b = c3_b,
            .s4_w = s4_w, .s4_b = s4_b,
            .c5_w = c5_w, .c5_b = c5_b,
            .f6_w = f6_w, .f6_b = f6_b,
            .rbf_templates = rbf_templates,
        }, protocol::SERVER);
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
        shark::span<u64> image(32 * 32);
        unsigned expected = 0;
        if (pid == protocol::CLIENT)
        {
            const auto sample = cnpy::npz_load(argv[2]);
            const auto pixels = read_array<u64>(sample, "image", {32, 32, 1});
            std::copy(pixels.begin(), pixels.end(), image.begin());
            expected = read_array<u8>(sample, "label", {1})[0];
        }

        if (pid == protocol::DEALER)
        {
            // Generate fresh preprocessing for each execution.
            std::random_device random;
            protocol::init::gen(std::uniform_int_distribution<u64>{}(random));
        }
        else
        {
            protocol::init::eval(pid, "127.0.0.1", 42069);
        }

        fs::lenet5 model;
        if (pid == protocol::SERVER)
            load_weights(model, argc == 3 ? argv[2] : "data/lenet5_weights.npz");
        else
            model.load_params({}, protocol::SERVER);

        protocol::input::call(image, protocol::CLIENT);
        auto prediction = model.inf(image);
        protocol::output::call(prediction);
        protocol::finalize::call();

        if (pid == protocol::CLIENT)
            std::cout << "prediction=" << prediction[0] << " label=" << expected << '\n';
    }
    catch (const std::exception& error)
    {
        std::cerr << "LeNet-5: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
