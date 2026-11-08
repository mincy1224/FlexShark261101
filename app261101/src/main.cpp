#include <shark/protocols/common.hpp>
#include <shark/protocols/init.hpp>
#include <shark/protocols/finalize.hpp>
#include <shark/protocols/input.hpp>
#include <shark/protocols/output.hpp>
#include <shark/protocols/matmul.hpp>

#include <iostream>
#include <random>

using namespace shark;
using namespace shark::protocols;

int main(int argc, char** argv)
{
    init::from_args(argc, argv);

    constexpr u64 m = 2;
    constexpr u64 k = 5;
    constexpr u64 n = 2;

    span<u64> A(m * k);
    span<u64> B(k * n);

    // 仅用于生成小整数测试输入，不用于生成协议随机掩码。
    std::mt19937_64 rng(std::random_device{}());
    std::uniform_int_distribution<unsigned> dist(1, 9);

    // A 的明文只由 Server 准备。
    if (party == SERVER) 
    {
        for (u64 i = 0; i < A.size(); ++i) 
        {
            A[i] = dist(rng);
        }
    }

    // B 的明文只由 Client 准备。
    if (party == CLIENT) 
    {
        for (u64 i = 0; i < B.size(); ++i) 
        {
            B[i] = dist(rng);
        }
    }

    // 三个角色都必须按同样的顺序调用。
    input::call(A, SERVER);
    input::call(B, CLIENT);

    auto C = matmul::call(m, k, n, A, B);

    // 检查并公开结果。当前接口向双方公开。
    output::call(C);

    if (party == SERVER) 
    {
        std::cout << "C = A * B:\n";
        for (u64 i = 0; i < m; ++i) 
        {
            for (u64 j = 0; j < n; ++j) 
            {
                std::cout << C[i * n + j] << " ";
            }
            
            std::cout << '\n';
        }
    }

    finalize::call();
    return 0;
}