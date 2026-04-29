#include <iostream>
#include "flashmla.cuh"
#include "online_softmax.cuh"
#include "fr-tensor.cuh"

int main() {
    // small dimensions
    uint m = 2, n = 2, d = 3;
    std::cout << "self-attn pipeline test" << std::endl;
    // random Q K V
    FrTensor Q(m*d), K(n*d), V(n*d);
    std::vector<Fr_t> hQ(m*d), hK(n*d), hV(n*d);
    for (uint i = 0; i < m*d; ++i) hQ[i] = {static_cast<uint>(i+1),0,0,0,0,0,0,0};
    for (uint i = 0; i < n*d; ++i) hK[i] = {static_cast<uint>(i+2),0,0,0,0,0,0,0};
    for (uint i = 0; i < n*d; ++i) hV[i] = {static_cast<uint>(i+3),0,0,0,0,0,0,0};
    cudaMemcpy(Q.gpu_data, hQ.data(), sizeof(Fr_t)*m*d, cudaMemcpyHostToDevice);
    cudaMemcpy(K.gpu_data, hK.data(), sizeof(Fr_t)*n*d, cudaMemcpyHostToDevice);
    cudaMemcpy(V.gpu_data, hV.data(), sizeof(Fr_t)*n*d, cudaMemcpyHostToDevice);

    // compute attention logits with flashmla
    FrTensor X(m*n);
    FlashMLAConfig cfg{2,2,d};
    FrTensor blk_max(0), blk_sum(0);
    allocate_flash_blocks(blk_max, blk_sum, m, n, cfg);
    flashmla_kernel(Q, K, X, cfg, blk_max, blk_sum);

    // run online softmax on X
    FrTensor shift(m);
    FrTensor X_shifted(m*n);
    vector<FrTensor> aux_states;
    OnlineSoftmaxCtx ctx{FrTensor(0),FrTensor(0),FrTensor(0),FrTensor(0)};
    // identity exp table
    int tableSize = 4;
    ctx.exp_table = FrTensor(tableSize);
    vector<Fr_t> htab(tableSize);
    for (int i=0;i<tableSize;i++) htab[i] = {static_cast<uint>(i),0,0,0,0,0,0,0};
    cudaMemcpy(ctx.exp_table.gpu_data, htab.data(), sizeof(Fr_t)*tableSize, cudaMemcpyHostToDevice);
    FrTensor Y = online_softmax_compute(X, ctx, shift, X_shifted, aux_states);

    // multiply Y * V (m x n times n x d)
    FrTensor out(m*d);
    // simple CPU matmul for demonstration
    std::vector<Fr_t> hY(m*n), hout(m*d);
    cudaMemcpy(hY.data(), Y.gpu_data, sizeof(Fr_t)*m*n, cudaMemcpyDeviceToHost);
    cudaMemcpy(hV.data(), V.gpu_data, sizeof(Fr_t)*n*d, cudaMemcpyDeviceToHost);
    for (uint i=0;i<m;i++){
        for(uint j=0;j<d;j++){
            Fr_t sum = {0,0,0,0,0,0,0,0};
            for(uint k=0;k<n;k++){
                // not using field arithmetic here just accumulate limb0 part
                uint64_t a = hY[i*n+k].val[0];
                uint64_t b = hV[k*d+j].val[0];
                sum.val[0] += a*b;
            }
            hout[i*d+j] = sum;
        }
    }
    cudaMemcpy(out.gpu_data, hout.data(), sizeof(Fr_t)*m*d, cudaMemcpyHostToDevice);
    std::cout<<"Self-attn pipeline finished. Output limb0:";
    for(uint i=0;i<m*d;i++){ std::cout<<hout[i].val[0]<<","; }
    std::cout<<std::endl;
    return 0;
}
