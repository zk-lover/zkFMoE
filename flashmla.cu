#include "flashmla.cuh"
#include "fr-tensor.cuh"
#include "ioutils.cuh"

// basic blocked attention kernel for FlashMLA
// Q: (m x d), K: (n x d) stored row-major; out: (m x n)
// cfg specifies block dimensions: Bq = block rows of Q, Bk = block rows of K, Bd = embedding dim
// block_max and block_sum are sized (m/Bq) x (n/Bk), storing per-block maximum and sum

__global__ void flashmla_block_kernel(const Fr_t* Qdata, const Fr_t* Kdata, Fr_t* outdata,
                                      Fr_t* maxdata, Fr_t* sumdata,
                                      uint m, uint n, uint d,
                                      FlashMLAConfig cfg)
{
    uint bi = blockIdx.y; // block row index (for Q)
    uint bj = blockIdx.x; // block column index (for K)
    uint ti = threadIdx.y;
    uint tj = threadIdx.x;
    uint Bq = cfg.Bq;
    uint Bk = cfg.Bk;

    uint row = bi * Bq + ti;
    uint col = bj * Bk + tj;

    if (row >= m || col >= n) return;

    // compute dot product for this output element
    Fr_t acc = {0,0,0,0,0,0,0,0};
    for (uint k = 0; k < d; ++k) {
        // Q[row,d] * K[col,d]
        // convert plain input to montgomery representation
        Fr_t q_plain = Qdata[row * d + k];
        Fr_t kk_plain = Kdata[col * d + k];
        Fr_t q = blstrs__scalar__Scalar_mont(q_plain);
        Fr_t kk = blstrs__scalar__Scalar_mont(kk_plain);
        Fr_t prod = blstrs__scalar__Scalar_mul(q, kk);
        acc = blstrs__scalar__Scalar_add(acc, prod);
    }
    // convert back from montgomery form for easier verification
    Fr_t acc_out = blstrs__scalar__Scalar_unmont(acc);
    outdata[row * n + col] = acc_out;

    // atomic track max and sum per block using unmonted value
    uint block_index = bi * ((n + Bk - 1)/Bk) + bj;
    unsigned long long aval = (unsigned long long)acc_out.val[0] | ((unsigned long long)acc_out.val[1] << 32);
    atomicMax((unsigned long long*)&maxdata[block_index], aval);
    atomicAdd((unsigned long long*)&sumdata[block_index], aval);
}

void flashmla_kernel(const FrTensor &Q, const FrTensor &K, FrTensor &out,
                     const FlashMLAConfig &cfg,
                     FrTensor &block_max, FrTensor &block_sum)
{
    uint m = Q.size / cfg.Bd;
    uint n = K.size / cfg.Bd;
    uint d = cfg.Bd;

    dim3 blockSize(16, 16);
    dim3 gridSize((n + cfg.Bk - 1)/cfg.Bk, (m + cfg.Bq - 1)/cfg.Bq);
    flashmla_block_kernel<<<gridSize, blockSize>>>(Q.gpu_data, K.gpu_data, out.gpu_data,
                                                     block_max.gpu_data, block_sum.gpu_data,
                                                     m, n, d, cfg);
    cudaDeviceSynchronize();
}

void flashmla_reduce(const FrTensor &partial, FrTensor &out)
{
    // placeholder: copy
    out = partial;
}
