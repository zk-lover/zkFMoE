#include "zkmoe.cuh"
#include "zkfc.cuh"
#include "tlookup.cuh"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {

KERNEL void scalar_to_int_host_kernel(const Fr_t* scalar_ptr, int* int_ptr, uint n)
{
    const uint gid = GET_GLOBAL_ID();
    if (gid >= n) return;
    int_ptr[gid] = scalar_to_int(scalar_ptr[gid]);
}

bool is_one(const Fr_t &x)
{
    return x.val[0] == 1 && x.val[1] == 0 && x.val[2] == 0 && x.val[3] == 0 &&
           x.val[4] == 0 && x.val[5] == 0 && x.val[6] == 0 && x.val[7] == 0;
}

bool is_zero(const Fr_t &x)
{
    return x.val[0] == 0 && x.val[1] == 0 && x.val[2] == 0 && x.val[3] == 0 &&
           x.val[4] == 0 && x.val[5] == 0 && x.val[6] == 0 && x.val[7] == 0;
}

FrTensor ints_to_frtensor(const std::vector<int> &vals)
{
    return FrTensor((uint)vals.size(), vals.data());
}

void frtensor_to_host_ints(const FrTensor &x, std::vector<int> &out)
{
    out.resize(x.size);
    int *d_int = nullptr;
    cudaMalloc((void **)&d_int, sizeof(int) * x.size);
    scalar_to_int_host_kernel<<<(x.size + FrNumThread - 1) / FrNumThread, FrNumThread>>>(x.gpu_data, d_int, x.size);
    cudaDeviceSynchronize();
    cudaMemcpy(out.data(), d_int, sizeof(int) * x.size, cudaMemcpyDeviceToHost);
    cudaFree(d_int);
}

} // namespace

zkMoE::zkMoE(uint E_, uint K_,
             const FrTensor &range_tbl_,
             const FrTensor &cmp_tbl_,
             const FrTensor &memb_tbl_,
             const FrTensor &exp_tbl_)
    : E(E_), K(K_), range_tbl(range_tbl_), cmp_tbl(cmp_tbl_), memb_tbl(memb_tbl_), exp_tbl(exp_tbl_)
{
}

pair<FrTensor, vector<Polynomial>> zkMoE::prep(const FrTensor &logits,
                                               const FrTensor &mask,
                                               vector<Polynomial> &proof)
{
    if (logits.size % E != 0) {
        throw std::runtime_error("zkMoE::prep size mismatch");
    }
    if (mask.size != E && mask.size != logits.size) {
        throw std::runtime_error("zkMoE::prep mask size mismatch");
    }

    uint T = logits.size / E;

    std::vector<int> logits_i, mask_i;
    frtensor_to_host_ints(logits, logits_i);
    frtensor_to_host_ints(mask, mask_i);

    std::vector<Fr_t> hout(logits.size, {0,0,0,0,0,0,0,0});
    for (uint t = 0; t < T; ++t) {
        std::vector<uint> valid;
        valid.reserve(E);
        for (uint i = 0; i < E; ++i) {
            int mask_val = (mask.size == E) ? mask_i[i] : mask_i[t * E + i];
            if (mask_val != 0) valid.push_back(i);
        }
        if (valid.size() < K) {
            throw std::runtime_error("zkMoE::prep not enough valid experts for top-k");
        }

        std::sort(valid.begin(), valid.end(), [&](uint a, uint b) {
            return logits_i[t * E + a] > logits_i[t * E + b];
        });

        for (uint j = 0; j < K; ++j) {
            hout[t * E + valid[j]] = {1,0,0,0,0,0,0,0};
        }
    }

    FrTensor out(logits.size, hout.data());
    return {out, proof};
}

Fr_t zkMoE::prove(const FrTensor &out,
                   const FrTensor &logits,
                   const FrTensor &expert_out,
                   const FrTensor &mask,
                   vector<Polynomial> &proof)
{
    if (logits.size % E != 0 || out.size != logits.size || expert_out.size != logits.size) {
        throw std::runtime_error("zkMoE::prove size mismatch");
    }
    if (mask.size != E && mask.size != logits.size) {
        throw std::runtime_error("zkMoE::prove mask size mismatch");
    }

    uint T = logits.size / E;

    std::vector<int> logits_i, expert_i, mask_i;
    frtensor_to_host_ints(logits, logits_i);
    frtensor_to_host_ints(expert_out, expert_i);
    frtensor_to_host_ints(mask, mask_i);

    std::vector<Fr_t> out_h(out.size);
    cudaMemcpy(out_h.data(), out.gpu_data, sizeof(Fr_t) * out.size, cudaMemcpyDeviceToHost);

    if (range_tbl.size == 0 || cmp_tbl.size == 0 || memb_tbl.size == 0) {
        throw std::runtime_error("zkMoE::prove lookup tables are empty");
    }

    std::vector<int> range_host, cmp_host;
    frtensor_to_host_ints(range_tbl, range_host);
    frtensor_to_host_ints(cmp_tbl, cmp_host);
    int range_low = range_host.front();
    int range_high = range_host.back();
    int range_len = range_high - range_low + 1;
    int cmp_low = cmp_host.front();
    int cmp_high = cmp_host.back();
    int cmp_len = cmp_high - cmp_low + 1;

    std::vector<int> tau_vals;
    tau_vals.reserve(T);

    std::vector<int> all_diffs;
    all_diffs.reserve(static_cast<size_t>(T) * static_cast<size_t>(K) * static_cast<size_t>(E - K));
    std::vector<int> all_cmp_diffs;
    all_cmp_diffs.reserve(static_cast<size_t>(T) * static_cast<size_t>(E));
    int min_diff = 0;
    int max_diff = 0;
    bool first_diff = true;

    for (uint t = 0; t < T; ++t) {
        std::vector<int> selected;
        selected.reserve(K);
        std::vector<int> unselected;
        unselected.reserve(E);

        for (uint i = 0; i < E; ++i) {
            int mask_val = (mask.size == E) ? mask_i[i] : mask_i[t * E + i];
            const Fr_t &sel = out_h[t * E + i];
            if (is_one(sel)) {
                if (mask_val == 0) {
                    throw std::runtime_error("zkMoE::prove selected index invalid or masked out");
                }
                selected.push_back((int)i);
            } else {
                if (mask_val != 0) unselected.push_back((int)i);
            }
        }

        if ((uint)selected.size() != K) {
            throw std::runtime_error("zkMoE::prove selected count mismatch with K");
        }

        int tau = logits_i[t * E + selected[0]];
        for (int sidx : selected) {
            tau = std::min(tau, logits_i[t * E + sidx]);
        }
        tau_vals.push_back(tau);

        for (uint i = 0; i < E; ++i) {
            int mask_val = (mask.size == E) ? mask_i[i] : mask_i[t * E + i];
            if (mask_val == 0) continue;
            int d = logits_i[t * E + i] - tau;
            all_cmp_diffs.push_back(d);
        }

        for (int s : selected) {
            for (int uidx : unselected) {
                int d = logits_i[t * E + s] - logits_i[t * E + uidx];
                all_diffs.push_back(d);
                if (first_diff) {
                    min_diff = max_diff = d;
                    first_diff = false;
                } else {
                    min_diff = std::min(min_diff, d);
                    max_diff = std::max(max_diff, d);
                }
            }
        }
    }

    {
        for (int value : logits_i) {
            if (value < range_low || value > range_high) {
                throw std::runtime_error("zkMoE::prove logits out of pre-generated range table");
            }
        }
        int *d_logits = nullptr;
        cudaMalloc((void **)&d_logits, sizeof(int) * logits.size);
        cudaMemcpy(d_logits, logits_i.data(), sizeof(int) * logits.size, cudaMemcpyHostToDevice);
        tLookupRange range_lookup(range_low, (uint)range_len);
        FrTensor m_range = range_lookup.prep(d_logits, logits.size);
        auto u = random_vec(ceilLog2(logits.size));
        auto v = random_vec(ceilLog2(logits.size));
        Fr_t alpha = {1,0,0,0,0,0,0,0};
        Fr_t beta = {1,0,0,0,0,0,0,0};
        range_lookup.prove(logits, m_range, alpha, beta, u, v, proof);
        cudaFree(d_logits);
    }

    {
        for (int tau : tau_vals) {
            if (tau < range_low || tau > range_high) {
                throw std::runtime_error("zkMoE::prove threshold out of range");
            }
        }
        int *d_tau = nullptr;
        cudaMalloc((void **)&d_tau, sizeof(int) * tau_vals.size());
        cudaMemcpy(d_tau, tau_vals.data(), sizeof(int) * tau_vals.size(), cudaMemcpyHostToDevice);

        tLookupRangeMapping memb_mapping(range_low, (uint)range_len, memb_tbl);
        auto pr_memb = memb_mapping(d_tau, (uint)tau_vals.size());
        FrTensor y_memb = pr_memb.first;
        FrTensor m_memb = pr_memb.second;
        auto u = random_vec(ceilLog2((uint)tau_vals.size()));
        auto v = random_vec(ceilLog2((uint)tau_vals.size()));
        auto rv = random_vec(1);
        Fr_t r = rv[0];
        Fr_t alpha = {1,0,0,0,0,0,0,0};
        Fr_t beta = {1,0,0,0,0,0,0,0};
        FrTensor tau_tensor = ints_to_frtensor(tau_vals);
        memb_mapping.prove(tau_tensor, y_memb, m_memb, r, alpha, beta, u, v, proof);
        Fr_t memb_sum = y_memb.sum();
        Fr_t t_fr = {(uint)tau_vals.size(), 0, 0, 0, 0, 0, 0, 0};
        if (memb_sum != t_fr) {
            cudaFree(d_tau);
            throw std::runtime_error("zkMoE::prove threshold membership check failed");
        }
        cudaFree(d_tau);
    }

    if (all_diffs.empty()) {
        return {(uint)(T * K), 0, 0, 0, 0, 0, 0, 0};
    }

    if (cmp_len <= 0) {
        throw std::runtime_error("zkMoE::prove invalid compare table range");
    }
    if (min_diff < cmp_low || max_diff > cmp_high) {
        throw std::runtime_error("zkMoE::prove diff out of pre-generated compare table range");
    }

    std::vector<int> mapped_cpu(cmp_len, 0);
    for (int v = 0; v < cmp_len; ++v) {
        mapped_cpu[v] = (cmp_low + v >= 0) ? 1 : 0;
    }

    FrTensor mapped_vals(cmp_len, mapped_cpu.data());
    tLookupRangeMapping mapping(cmp_low, (uint)cmp_len, mapped_vals);

    int M = (int)all_diffs.size();
    int *d_vals = nullptr;
    cudaMalloc((void **)&d_vals, sizeof(int) * M);
    cudaMemcpy(d_vals, all_diffs.data(), sizeof(int) * M, cudaMemcpyHostToDevice);

    FrTensor diffs_tensor(M, all_diffs.data());
    auto pr = mapping(d_vals, (uint)M);
    FrTensor y = pr.first;
    FrTensor m = pr.second;

    uint lg = ceilLog2((uint)M);
    auto u = random_vec(lg);
    auto v = random_vec(lg);
    auto rv = random_vec(1);
    Fr_t r = rv[0];
    Fr_t alpha = {1,0,0,0,0,0,0,0};
    Fr_t beta = {1,0,0,0,0,0,0,0};

    mapping.prove(diffs_tensor, y, m, r, alpha, beta, u, v, proof);

    Fr_t count = y.sum();
    Fr_t k_fr = {(uint)M, 0, 0, 0, 0, 0, 0, 0};
    bool k_ok = true;
    for (int i = 0; i < 8; ++i) {
        if (count.val[i] != k_fr.val[i]) {
            k_ok = false;
            break;
        }
    }
    if (!k_ok) {
        cudaFree(d_vals);
        throw std::runtime_error("zkMoE::prove pairwise mapped count mismatch");
    }

    {
        if (all_cmp_diffs.empty()) {
            cudaFree(d_vals);
            return count;
        }
        int cmp_min = all_cmp_diffs[0], cmp_max = all_cmp_diffs[0];
        for (int d : all_cmp_diffs) {
            cmp_min = std::min(cmp_min, d);
            cmp_max = std::max(cmp_max, d);
        }
        if (cmp_min < cmp_low || cmp_max > cmp_high) {
            cudaFree(d_vals);
            throw std::runtime_error("zkMoE::prove compare consistency out of cmp table range");
        }

        int *d_cmp = nullptr;
        cudaMalloc((void **)&d_cmp, sizeof(int) * all_cmp_diffs.size());
        cudaMemcpy(d_cmp, all_cmp_diffs.data(), sizeof(int) * all_cmp_diffs.size(), cudaMemcpyHostToDevice);
        tLookupRangeMapping cmp_mapping(cmp_low, (uint)cmp_len, ints_to_frtensor(mapped_cpu));
        auto pr_cmp = cmp_mapping(d_cmp, (uint)all_cmp_diffs.size());
        FrTensor y_cmp = pr_cmp.first;
        FrTensor m_cmp = pr_cmp.second;
        auto u = random_vec(ceilLog2((uint)all_cmp_diffs.size()));
        auto v = random_vec(ceilLog2((uint)all_cmp_diffs.size()));
        auto rv = random_vec(1);
        Fr_t r = rv[0];
        Fr_t alpha = {1,0,0,0,0,0,0,0};
        Fr_t beta = {1,0,0,0,0,0,0,0};
        FrTensor cmp_tensor = ints_to_frtensor(all_cmp_diffs);
        cmp_mapping.prove(cmp_tensor, y_cmp, m_cmp, r, alpha, beta, u, v, proof);

        size_t idx = 0;
        for (uint t = 0; t < T; ++t) {
            uint token_count_cmp = 0;
            uint token_count_sel = 0;
            for (uint i = 0; i < E; ++i) {
                int mask_val = (mask.size == E) ? mask_i[i] : mask_i[t * E + i];
                if (mask_val == 0) continue;
                Fr_t z_cmp = y_cmp((uint)idx);
                bool z_should = is_one(out_h[t * E + i]);
                if (z_should && is_zero(z_cmp)) {
                    cudaFree(d_cmp);
                    cudaFree(d_vals);
                    throw std::runtime_error("zkMoE::prove compare/z subset consistency mismatch");
                }
                if (is_one(z_cmp)) token_count_cmp++;
                if (z_should) token_count_sel++;
                idx++;
            }
            if (token_count_sel != K || token_count_cmp < K) {
                cudaFree(d_cmp);
                cudaFree(d_vals);
                throw std::runtime_error("zkMoE::prove compare-based top-k count lower-bound mismatch");
            }
        }
        cudaFree(d_cmp);
    }

    {
        std::vector<int> exp_table;
        frtensor_to_host_ints(exp_tbl, exp_table);
        if (exp_table.empty()) {
            cudaFree(d_vals);
            throw std::runtime_error("zkMoE::prove exp table is empty");
        }

        std::vector<int> exp_diffs;
        std::vector<int> exp_vals;
        exp_diffs.reserve(logits.size);
        exp_vals.reserve(logits.size);
        int exp_low = cmp_low;
        int exp_high = exp_low + (int)exp_table.size() - 1;

        for (uint t = 0; t < T; ++t) {
            int tau = tau_vals[t];
            for (uint i = 0; i < E; ++i) {
                int d = logits_i[t * E + i] - tau;
                if (d < exp_low || d > exp_high) {
                    cudaFree(d_vals);
                    throw std::runtime_error("zkMoE::prove exp diff out of exp table range");
                }
                exp_diffs.push_back(d);
                exp_vals.push_back(exp_table[d - exp_low]);
            }
        }

        int *d_exp = nullptr;
        cudaMalloc((void **)&d_exp, sizeof(int) * exp_diffs.size());
        cudaMemcpy(d_exp, exp_diffs.data(), sizeof(int) * exp_diffs.size(), cudaMemcpyHostToDevice);
        tLookupRangeMapping exp_mapping(exp_low, (uint)exp_table.size(), exp_tbl);
        auto pr_exp = exp_mapping(d_exp, (uint)exp_diffs.size());
        FrTensor y_exp = pr_exp.first;
        FrTensor m_exp = pr_exp.second;
        auto u = random_vec(ceilLog2((uint)exp_diffs.size()));
        auto v = random_vec(ceilLog2((uint)exp_diffs.size()));
        auto rv = random_vec(1);
        Fr_t r = rv[0];
        Fr_t alpha = {1,0,0,0,0,0,0,0};
        Fr_t beta = {1,0,0,0,0,0,0,0};
        FrTensor exp_diff_tensor = ints_to_frtensor(exp_diffs);
        exp_mapping.prove(exp_diff_tensor, y_exp, m_exp, r, alpha, beta, u, v, proof);

        std::vector<int> y_exp_host;
        frtensor_to_host_ints(y_exp, y_exp_host);
        for (size_t i = 0; i < exp_vals.size(); ++i) {
            int yv = y_exp_host[i];
            if (yv != exp_vals[i]) {
                cudaFree(d_exp);
                cudaFree(d_vals);
                throw std::runtime_error("zkMoE::prove exp lookup mismatch");
            }
        }

        for (uint t = 0; t < T; ++t) {
            long long denom = 0;
            long long numer = 0;
            for (uint i = 0; i < E; ++i) {
                int z = is_one(out_h[t * E + i]) ? 1 : 0;
                int e = exp_vals[t * E + i];
                int eo = expert_i[t * E + i];
                long long w_num = (long long)z * (long long)e;
                denom += w_num;
                numer += w_num * (long long)eo;
            }
            if (denom == 0) {
                cudaFree(d_exp);
                cudaFree(d_vals);
                throw std::runtime_error("zkMoE::prove zero denominator in proxy forward consistency");
            }
            (void)numer;
        }
        cudaFree(d_exp);
    }

    cudaFree(d_vals);
    return count;
}

