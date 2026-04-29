#pragma once

#include "fr-tensor.cuh"

// Configuration for FlashMLA blocking
struct FlashMLAConfig {
    uint Bq;    // block size for queries
    uint Bk;    // block size for keys
    uint Bd;    // inner dimension block size (usually embedding dimension)
};

// kernels defined in flashmla.cu
void flashmla_kernel(const FrTensor &Q, const FrTensor &K, FrTensor &out,
                     const FlashMLAConfig &cfg,
                     FrTensor &block_max, FrTensor &block_sum);

// reduce partial results across blocks (if needed)
void flashmla_reduce(const FrTensor &partial, FrTensor &out);

// helper to allocate block result tensors based on m,n and cfg
inline void allocate_flash_blocks(FrTensor &blk_max, FrTensor &blk_sum,
                                  uint m, uint n, const FlashMLAConfig &cfg) {
    uint rows = (m + cfg.Bq - 1) / cfg.Bq;
    uint cols = (n + cfg.Bk - 1) / cfg.Bk;
    blk_max = FrTensor(rows * cols);
    blk_sum = FrTensor(rows * cols);
    cudaMemset(blk_max.gpu_data, 0, blk_max.size * sizeof(Fr_t));
    cudaMemset(blk_sum.gpu_data, 0, blk_sum.size * sizeof(Fr_t));
}
