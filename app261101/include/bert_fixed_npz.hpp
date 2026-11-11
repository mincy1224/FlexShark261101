#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <initializer_list>
#include <span>
#include <stdexcept>
#include <string>

#include <cnpy.h>
#include <shark/types/u64.hpp>
#include <shark/utils/globals.hpp>

namespace fs::bert_fixed_npz
{
    inline constexpr std::size_t LAYER_COUNT = 12;
    inline constexpr std::size_t HEAD_COUNT = 12;
    inline constexpr std::size_t HIDDEN_SIZE = 768;
    inline constexpr std::size_t FFN_SIZE = 3072;
    inline constexpr std::size_t LABEL_COUNT = 2;
    inline constexpr std::size_t TOKEN_COUNT = 128;

#ifdef FS_BERT_DEFAULT_WEIGHTS
    inline constexpr const char* DEFAULT_WEIGHTS = FS_BERT_DEFAULT_WEIGHTS;
#else
    inline constexpr const char* DEFAULT_WEIGHTS = "bert_data/cache/bert_weights_fixed.npz";
#endif

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

    inline cnpy::npz_t load_archive(const char* path)
    {
        const std::array<shark::u64, 10> expected_config = {
            1, LAYER_COUNT, HEAD_COUNT, HIDDEN_SIZE, FFN_SIZE, LABEL_COUNT,
            TOKEN_COUNT, FLOAT_PRECISION_64, FLOAT_PRECISION_32, FLOAT_PRECISION_16
        };

        auto archive = cnpy::npz_load(path);
        if (archive.empty())
            throw std::runtime_error(std::string("empty or invalid NPZ archive: ") + path);
        if (archive.contains("config"))
        {
            const auto config = read_array<shark::u64>(archive, "config", {expected_config.size()});
            if (!std::equal(config.begin(), config.end(), expected_config.begin()))
                throw std::runtime_error(std::string("incompatible BERT NPZ config: ") + path);
        }
        return archive;
    }
}
