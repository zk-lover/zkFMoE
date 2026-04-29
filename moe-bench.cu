#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include "zkmoe.cuh"
#include "fr-tensor.cuh"
#include "timer.hpp"

using namespace std;

namespace {
KERNEL void scalar_to_int_host_kernel_local(const Fr_t* scalar_ptr, int* int_ptr, uint n)
{
    const uint gid = GET_GLOBAL_ID();
    if (gid >= n) return;
    int_ptr[gid] = scalar_to_int(scalar_ptr[gid]);
}

void frtensor_to_host_ints_local(const FrTensor &x, std::vector<int> &out)
{
    out.resize(x.size);
    int *d_int = nullptr;
    cudaMalloc((void **)&d_int, sizeof(int) * x.size);
    scalar_to_int_host_kernel_local<<<(x.size + FrNumThread - 1) / FrNumThread, FrNumThread>>>(x.gpu_data, d_int, x.size);
    cudaDeviceSynchronize();
    cudaMemcpy(out.data(), d_int, sizeof(int) * x.size, cudaMemcpyDeviceToHost);
    cudaFree(d_int);
}
}

int main(int argc, char* argv[])
{
    if (argc < 6) {
        cerr << "usage: moe-bench <logits_int_bin> <expert_out_int_bin> <seq_len> <num_experts> <top_k>\n";
        return 1;
    }

    string logits_file = argv[1];
    string expert_file = argv[2];
    uint seq_len = static_cast<uint>(stoi(argv[3]));
    uint num_experts = static_cast<uint>(stoi(argv[4]));
    uint top_k = static_cast<uint>(stoi(argv[5]));

    FrTensor logits_all = FrTensor::from_int_bin(logits_file);
    FrTensor expert_all = FrTensor::from_int_bin(expert_file);
    if (logits_all.size != seq_len * num_experts) {
        cerr << "moe-bench: logits size mismatch, got " << logits_all.size
             << " expected " << (seq_len * num_experts) << "\n";
        return 2;
    }
    if (expert_all.size != seq_len * num_experts) {
        cerr << "moe-bench: expert_out size mismatch, got " << expert_all.size
             << " expected " << (seq_len * num_experts) << "\n";
        return 3;
    }

    vector<Fr_t> h_mask(num_experts, {1,0,0,0,0,0,0,0});
    FrTensor mask_row(num_experts, h_mask.data());

    vector<int> logits_i;
    frtensor_to_host_ints_local(logits_all, logits_i);

    int min_logit = logits_i[0], max_logit = logits_i[0];
    for (int value : logits_i) {
        min_logit = std::min(min_logit, value);
        max_logit = std::max(max_logit, value);
    }
    int span = max_logit - min_logit;

    vector<int> range_vals(span + 1);
    for (int i = 0; i <= span; ++i) range_vals[i] = min_logit + i;
    FrTensor range_tbl((uint)range_vals.size(), range_vals.data());

    int cmp_low = -span;
    int cmp_high = span;
    vector<int> cmp_vals(cmp_high - cmp_low + 1);
    for (int i = 0; i < (int)cmp_vals.size(); ++i) cmp_vals[i] = cmp_low + i;
    FrTensor cmp_tbl((uint)cmp_vals.size(), cmp_vals.data());

    vector<int> present(span + 1, 0);
    for (int value : logits_i) present[value - min_logit] = 1;
    FrTensor memb_tbl((uint)present.size(), present.data());

    int exp_low = cmp_low;
    int exp_high = cmp_high;
    const int exp_scale = 1024;
    vector<int> exp_vals(exp_high - exp_low + 1, 0);
    for (int i = 0; i < (int)exp_vals.size(); ++i) {
        int d = exp_low + i;
        double ev = std::exp((double)d / 1024.0);
        long long q = llround(ev * (double)exp_scale);
        if (q < 0) q = 0;
        if (q > 2147483647LL) q = 2147483647LL;
        exp_vals[i] = (int)q;
    }
    FrTensor exp_tbl((uint)exp_vals.size(), exp_vals.data());

    zkMoE moe(num_experts, top_k, range_tbl, cmp_tbl, memb_tbl, exp_tbl);

    Timer timer;
    vector<Polynomial> proof;
    timer.start();
    auto p = moe.prep(logits_all, mask_row, proof);
    FrTensor out = p.first;
    (void)moe.prove(out, logits_all, expert_all, mask_row, proof);
    timer.stop();

    size_t proof_poly_count = proof.size();
    size_t proof_coeff_count = 0;
    for (auto &poly : proof) {
        proof_coeff_count += static_cast<size_t>(poly.getDegree() + 1);
    }
    size_t proof_est_bytes = proof_coeff_count * sizeof(Fr_t);
    size_t pairwise_constraints = static_cast<size_t>(seq_len) * static_cast<size_t>(top_k) * static_cast<size_t>(num_experts - top_k);

    cout << "=== MoE benchmark results ===\n";
    cout << "seq_len           : " << seq_len << "\n";
    cout << "num_experts       : " << num_experts << "\n";
    cout << "top_k             : " << top_k << "\n";
    cout << "pairwise_checks   : " << pairwise_constraints << "\n";
    cout << "moe_total_time_s  : " << timer.getTotalTime() << "\n";
    cout << "verifier_time_s   : 0\n";
    cout << "PROOF_STATS_MOE poly_count=" << proof_poly_count
         << " coeff_count=" << proof_coeff_count
         << " est_bytes=" << proof_est_bytes << "\n";

    return 0;
}
