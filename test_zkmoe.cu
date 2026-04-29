#include <iostream>
#include "zkmoe.cuh"
#include "fr-tensor.cuh"

int main() {
    const uint E = 4;
    const uint K = 2;
    // dummy lookup tables, not used by naive prep
    FrTensor range_tbl(1), cmp_tbl(1), memb_tbl(1);
    zkMoE model(E, K, range_tbl, cmp_tbl, memb_tbl);

    FrTensor logits(E);
    FrTensor mask(E);
    std::vector<Fr_t> hlog(E), hmask(E);
    for (uint i = 0; i < E; ++i) {
        hlog[i] = {static_cast<uint>(i+1),0,0,0,0,0,0,0};
        hmask[i] = {1,0,0,0,0,0,0,0};
    }
    cudaMemcpy(logits.gpu_data, hlog.data(), sizeof(Fr_t)*E, cudaMemcpyHostToDevice);
    cudaMemcpy(mask.gpu_data, hmask.data(), sizeof(Fr_t)*E, cudaMemcpyHostToDevice);

    std::vector<Polynomial> proof;
    auto pr = model.prep(logits, mask, proof);
    FrTensor out = pr.first;

    std::vector<Fr_t> hout(E);
    cudaMemcpy(hout.data(), out.gpu_data, sizeof(Fr_t)*E, cudaMemcpyDeviceToHost);
    std::cout << "top-k one-hot output:\n";
    for (uint i = 0; i < E; ++i) std::cout << hout[i].val[0] << ",";
    std::cout << std::endl;

    Fr_t res = model.prove(out, logits, mask, proof);
    std::cout << "proof result limb0=" << res.val[0] << std::endl;
    return 0;
}
