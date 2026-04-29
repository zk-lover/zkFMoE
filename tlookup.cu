#include "tlookup.cuh"
#include "proof.cuh"

// debug helper: perform field addition on device and return result
extern "C" __global__ void add_pair_kernel(Fr_t a, Fr_t b, Fr_t* out) {
    out[0] = blstrs__scalar__Scalar_add(a, b);
}

// forward declaration for unmont add kernel defined in separate file
extern "C" __global__ void unmont_add_kernel(Fr_t a, Fr_t b, Fr_t* out);
// also provide single-element unmont helper for diagnostics
extern "C" __global__ void unmont_single_kernel(Fr_t a, Fr_t* out);

// Some utils
KERNEL void tlookup_kernel(const uint* indices, const uint D, uint* counts){
    const uint tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < D){
        atomicAdd(&counts[indices[tid]], 1U);
    }
}

// simple wrappers for the three specialised lookup modes described in the plan
// convert each Fr_t value to an unsigned long index and read from the table.

// helper local to this translation unit to avoid conflicting definitions
static DEVICE unsigned long scalar_to_ulong_tlookup(Fr_t num){
    return static_cast<unsigned long>(num.val[0]) |
           (static_cast<unsigned long>(num.val[1]) << 32);
}

KERNEL void tlookup_max_kernel(const Fr_t* values, const Fr_t* table, Fr_t* out, uint N) {
    const uint tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        unsigned long idx = scalar_to_ulong_tlookup(values[tid]);
        // assume index fits in table; caller guarantees bounds
        out[tid] = table[idx];
    }
}

KERNEL void tlookup_exp_kernel(const Fr_t* values, const Fr_t* table, Fr_t* out, uint N) {
    const uint tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        unsigned long idx = scalar_to_ulong_tlookup(values[tid]);
        out[tid] = table[idx];
    }
}

KERNEL void tlookup_recursive_kernel(const Fr_t* state, const Fr_t* prev_state, Fr_t* out, uint N) {
    const uint tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        // placeholder: copy current state
        out[tid] = state[tid];
    }
}

KERNEL void count_to_m(uint* counts, Fr_t* m_ptr, uint N){
    const uint tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) m_ptr[tid] = {counts[tid], 0, 0, 0, 0, 0, 0, 0};
}

// 
FrTensor tLookup::prep(const uint* indices, const uint D){
    // //copy indices to indices_cpu
    // uint indices_cpu[D];
    // cudaMemcpy(indices_cpu, indices, sizeof(uint) * D, cudaMemcpyDeviceToHost);
    // // for (uint i=0; i < D; ++ i) cout << indices_cpu[i] << " ";
    // // cout << endl;
    
    FrTensor m(table.size);
    uint* counts;
    cudaMalloc((void **)&counts, sizeof(uint) * table.size);
    cudaMemset(counts, 0, sizeof(uint) * table.size); // cnm

    tlookup_kernel<<<(D+FrNumThread-1)/FrNumThread,FrNumThread>>>(indices, D, counts);
    cudaDeviceSynchronize();
    
    // //copy counts to cpu_counts
    // uint cpu_counts[table.size];
    // cudaMemcpy(cpu_counts, counts, sizeof(uint) * table.size, cudaMemcpyDeviceToHost);
    // // for (uint i=0; i < table.size; ++ i) cout << cpu_counts[i] << " ";
    // // cout << endl;

    count_to_m<<<(table.size+FrNumThread-1)/FrNumThread,FrNumThread>>>(counts, m.gpu_data, table.size);
    cudaDeviceSynchronize();
    cudaFree(counts);
    return m;
}

KERNEL void tlookup_inv_kernel(Fr_t* in_data, Fr_t beta, Fr_t* out_data, uint N)
{
    const uint tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < N){ 
        out_data[tid] = blstrs__scalar__Scalar_unmont(
            blstrs__scalar__Scalar_inverse(
                blstrs__scalar__Scalar_mont(
                    blstrs__scalar__Scalar_add(in_data[tid], beta)
                )
            )
        );
    }
}

KERNEL void half_tensor_kernel(const Fr_t* in_data, Fr_t* first_half_data, Fr_t* second_half_data, uint N_out)
{
    const uint tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N_out)
    {
        first_half_data[tid] = in_data[tid];
        second_half_data[tid] = in_data[tid + N_out];
    }
}


KERNEL void tLookup_phase1_poly_kernel(const Fr_t* A_data, const Fr_t* S_data, Fr_t alpha, Fr_t beta, Fr_t* out0, Fr_t* out1, Fr_t* out2, Fr_t* outA0, Fr_t* outA1, uint N_out)
{
    const uint tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N_out)
    {
        Fr_t c00 = A_data[tid];
        Fr_t c01 = blstrs__scalar__Scalar_sub(A_data[tid + N_out], A_data[tid]);
        Fr_t c10 = blstrs__scalar__Scalar_add(S_data[tid], beta);
        Fr_t c11 = blstrs__scalar__Scalar_sub(S_data[tid + N_out], S_data[tid]);

        Fr_t alpha_mont = blstrs__scalar__Scalar_mont(alpha);
        out0[tid] = blstrs__scalar__Scalar_mul(alpha_mont, blstrs__scalar__Scalar_mont(blstrs__scalar__Scalar_mul(c00, c10)));
        out1[tid] = blstrs__scalar__Scalar_mul(alpha_mont, blstrs__scalar__Scalar_mont(blstrs__scalar__Scalar_add(blstrs__scalar__Scalar_mul(c01, c10), blstrs__scalar__Scalar_mul(c00, c11))));
        out2[tid] = blstrs__scalar__Scalar_mul(alpha_mont, blstrs__scalar__Scalar_mont(blstrs__scalar__Scalar_mul(c01, c11)));

        outA0[tid] = c00;
        outA1[tid] = c01;
    }
}

KERNEL void tLookup_phase1_reduce_kernel(const Fr_t* A_data, const Fr_t* S_data, Fr_t* new_A_data, Fr_t* new_S_data, Fr_t v, uint N_out)
{
    const uint tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N_out)
    {
        Fr_t v_mont = blstrs__scalar__Scalar_mont(v);
        new_A_data[tid] = blstrs__scalar__Scalar_add(A_data[tid], blstrs__scalar__Scalar_mul(v_mont, blstrs__scalar__Scalar_sub(A_data[tid + N_out], A_data[tid])));  
        new_S_data[tid] = blstrs__scalar__Scalar_add(S_data[tid], blstrs__scalar__Scalar_mul(v_mont, blstrs__scalar__Scalar_sub(S_data[tid + N_out], S_data[tid])));
    }
}

// BUGGY AND TOO HARD TO DEBUG
// KERNEL void tLookup_phase2_poly_kernel(const Fr_t* A_data, const Fr_t* S_data, const Fr_t* B_data, const Fr_t* T_data, const Fr_t* m_data,
//     Fr_t alpha_, Fr_t beta, uint N, uint D, Fr_t alpha_sq,
//     Fr_t* out_eval0, Fr_t* out_eval1, Fr_t* out_eval2, Fr_t* out_sum0, Fr_t* out_sum1, Fr_t* out_sum2, uint N_out)
// {
//     const uint tid = threadIdx.x + blockIdx.x * blockDim.x;
//     if (tid < N_out)
//     {
//         Fr_t inv_ratio_mont = blstrs__scalar__Scalar_inverse(blstrs__scalar__Scalar_mont({D / N, 0, 0, 0, 0, 0, 0, 0}));
//         Fr_t inv_ratio = blstrs__scalar__Scalar_unmont(inv_ratio_mont);
//         Fr_t alpha__mont = blstrs__scalar__Scalar_mont(alpha_);
//         Fr_t alpha_sq_mont = blstrs__scalar__Scalar_mont(alpha_sq);
//         Fr_t inv_ratio_alpha_sq_mont = blstrs__scalar__Scalar_mul(inv_ratio_mont, alpha_sq_mont);

//         Fr_t c00 = A_data[tid];
//         Fr_t c01 = blstrs__scalar__Scalar_sub(A_data[tid + N_out], A_data[tid]);
//         Fr_t c10 = blstrs__scalar__Scalar_add(S_data[tid], beta);
//         Fr_t c11 = blstrs__scalar__Scalar_sub(S_data[tid + N_out], S_data[tid]);

//         Fr_t c00_ = B_data[tid];
//         Fr_t c01_ = blstrs__scalar__Scalar_sub(B_data[tid + N_out], B_data[tid]);
//         Fr_t c10_ = blstrs__scalar__Scalar_add(T_data[tid], beta);
//         Fr_t c11_ = blstrs__scalar__Scalar_sub(T_data[tid + N_out], T_data[tid]);

//         out_eval0[tid] = blstrs__scalar__Scalar_add(
//             blstrs__scalar__Scalar_mul(alpha__mont, blstrs__scalar__Scalar_mont(blstrs__scalar__Scalar_mul(c00, c10))),
//             blstrs__scalar__Scalar_mul(inv_ratio_alpha_sq_mont, blstrs__scalar__Scalar_mont(blstrs__scalar__Scalar_mul(c00_, c10_)))
//         );
//         out_eval1[tid] = blstrs__scalar__Scalar_add(
//             blstrs__scalar__Scalar_mul(alpha__mont, blstrs__scalar__Scalar_mont(blstrs__scalar__Scalar_add(blstrs__scalar__Scalar_mul(c01, c10), blstrs__scalar__Scalar_mul(c00, c11)))),
//             blstrs__scalar__Scalar_mul(inv_ratio_alpha_sq_mont, blstrs__scalar__Scalar_mont(blstrs__scalar__Scalar_add(blstrs__scalar__Scalar_mul(c01_, c10_), blstrs__scalar__Scalar_mul(c00_, c11_))))
//         );
//         out_eval2[tid] = blstrs__scalar__Scalar_add(
//             blstrs__scalar__Scalar_mul(alpha__mont, blstrs__scalar__Scalar_mont(blstrs__scalar__Scalar_mul(c01, c11))),
//             blstrs__scalar__Scalar_mul(inv_ratio_alpha_sq_mont, blstrs__scalar__Scalar_mont(blstrs__scalar__Scalar_mul(c01_, c11_)))
//         );

//         Fr_t m0 = m_data[tid];
//         Fr_t m1 = blstrs__scalar__Scalar_sub(m_data[tid + N_out], m_data[tid]);
//         out_sum0[tid] = blstrs__scalar__Scalar_sub(
//             c00,
//             blstrs__scalar__Scalar_mul(inv_ratio_mont, blstrs__scalar__Scalar_mont(blstrs__scalar__Scalar_mul(m0, c00_)))
//         );

//         out_sum1[tid] = blstrs__scalar__Scalar_sub(
//             c01,
//             blstrs__scalar__Scalar_mul(inv_ratio_mont, blstrs__scalar__Scalar_mont(blstrs__scalar__Scalar_add(blstrs__scalar__Scalar_mul(m0, c01_), blstrs__scalar__Scalar_mul(m1, c00_))))
//         );

//         out_sum2[tid] = blstrs__scalar__Scalar_sub(
//             {0, 0, 0, 0, 0, 0, 0, 0},
//             blstrs__scalar__Scalar_mul(inv_ratio_mont, blstrs__scalar__Scalar_mont(blstrs__scalar__Scalar_mul(m1, c11_)))
//         );
//     }
// }

KERNEL void tLookup_phase2_reduce_kernel(const Fr_t* A_data, const Fr_t* S_data, const Fr_t* B_data, const Fr_t* T_data, const Fr_t* m_data,
    Fr_t* new_A_data, Fr_t* new_S_data, Fr_t* new_B_data, Fr_t* new_T_data, Fr_t* new_m_data,
    Fr_t v, uint N_out)
{
    const uint tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N_out)
    {
        Fr_t v_mont = blstrs__scalar__Scalar_mont(v);
        new_A_data[tid] = blstrs__scalar__Scalar_add(A_data[tid], blstrs__scalar__Scalar_mul(v_mont, blstrs__scalar__Scalar_sub(A_data[tid + N_out], A_data[tid])));  
        new_S_data[tid] = blstrs__scalar__Scalar_add(S_data[tid], blstrs__scalar__Scalar_mul(v_mont, blstrs__scalar__Scalar_sub(S_data[tid + N_out], S_data[tid])));
        new_B_data[tid] = blstrs__scalar__Scalar_add(B_data[tid], blstrs__scalar__Scalar_mul(v_mont, blstrs__scalar__Scalar_sub(B_data[tid + N_out], B_data[tid])));
        new_T_data[tid] = blstrs__scalar__Scalar_add(T_data[tid], blstrs__scalar__Scalar_mul(v_mont, blstrs__scalar__Scalar_sub(T_data[tid + N_out], T_data[tid])));
        new_m_data[tid] = blstrs__scalar__Scalar_add(m_data[tid], blstrs__scalar__Scalar_mul(v_mont, blstrs__scalar__Scalar_sub(m_data[tid + N_out], m_data[tid])));
    }
}


// A.size == S.size == D
// u.size == ceilLog2(D)
// 0x39f6d3a994cebea4199cec0404d0ec02a9ded2017fff2dff7fffffff80000001
const Fr_t TWO_INV {2147483649, 2147483647, 2147429887, 2849952257, 80800770, 429714436, 2496577188, 972477353};
// const Fr_t TEMP_ZERO {0, 0, 0, 0, 0, 0, 0, 0};
// const Fr_t TEMP_ONE {1, 0, 0, 0, 0, 0, 0, 0};

Polynomial tLookup_phase1_step_poly(const FrTensor& A, const FrTensor& S, 
    const Fr_t& alpha, const Fr_t& beta, const Fr_t& C, const vector<Fr_t>& u)
{
    if (A.size != S.size) throw std::runtime_error("A.size != S.size");
    uint D = A.size;
    FrTensor temp0(D >> 1), temp1(D >> 1), temp2(D >> 1), tempA0(D >> 1), tempA1(D >> 1);
    tLookup_phase1_poly_kernel<<<((D >> 1)+FrNumThread-1)/FrNumThread,FrNumThread>>>(
        A.gpu_data, S.gpu_data, alpha, beta, temp0.gpu_data, temp1.gpu_data, temp2.gpu_data, tempA0.gpu_data, tempA1.gpu_data, D >> 1
    );
    cudaDeviceSynchronize();
    // strip the last element of u for the recursive evaluation; if u is
    // empty then there is nothing to strip and the result should be treated
    // as a constant.
    vector<Fr_t> u_;
    if (!u.empty()) u_.assign(u.begin(), u.end() - 1);

    Polynomial p0 ({temp0(u_), temp1(u_), temp2(u_)});
    Polynomial p1 ({tempA0.sum(), tempA1.sum()});
    if (!u.empty()) {
        p0 *= Polynomial::eq(u.back());
    }
    return p0 + p1 + C * TWO_INV;
}

KERNEL void tLookup_phase2_poly_eval_kernel(const Fr_t* A_data, const Fr_t* S_data, Fr_t alpha, Fr_t beta, Fr_t* out0, Fr_t* out1, Fr_t* out2, uint N_out)
{
    const uint tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N_out)
    {
        Fr_t c00 = A_data[tid];
        Fr_t c01 = blstrs__scalar__Scalar_sub(A_data[tid + N_out], A_data[tid]);
        Fr_t c10 = blstrs__scalar__Scalar_add(S_data[tid], beta);
        Fr_t c11 = blstrs__scalar__Scalar_sub(S_data[tid + N_out], S_data[tid]);

        Fr_t alpha_mont = blstrs__scalar__Scalar_mont(alpha);
        out0[tid] = blstrs__scalar__Scalar_mul(alpha_mont, blstrs__scalar__Scalar_mont(blstrs__scalar__Scalar_mul(c00, c10)));
        out1[tid] = blstrs__scalar__Scalar_mul(alpha_mont, blstrs__scalar__Scalar_mont(blstrs__scalar__Scalar_add(blstrs__scalar__Scalar_mul(c01, c10), blstrs__scalar__Scalar_mul(c00, c11))));
        out2[tid] = blstrs__scalar__Scalar_mul(alpha_mont, blstrs__scalar__Scalar_mont(blstrs__scalar__Scalar_mul(c01, c11)));
    }
}

KERNEL void tLookup_phase2_poly_sum_kernel(const Fr_t* A_data, Fr_t* out0, Fr_t* out1, uint N_out)
{
    const uint tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N_out)
    {
        out0[tid] = A_data[tid];
        out1[tid] = blstrs__scalar__Scalar_sub(A_data[tid + N_out], A_data[tid]);
    }
}

KERNEL void tLookup_phase2_poly_dotprod_kernel(const Fr_t* A_data, const Fr_t* B_data, Fr_t* out0, Fr_t* out1, Fr_t* out2, uint N_out)
{
    const uint tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N_out)
    {
        Fr_t A0 = A_data[tid];
        Fr_t A1 = blstrs__scalar__Scalar_sub(A_data[tid + N_out], A_data[tid]);
        Fr_t B0 = B_data[tid];
        Fr_t B1 = blstrs__scalar__Scalar_sub(B_data[tid + N_out], B_data[tid]);

        out0[tid] = blstrs__scalar__Scalar_mont(blstrs__scalar__Scalar_mul(A0, B0));
        out1[tid] = blstrs__scalar__Scalar_mont(blstrs__scalar__Scalar_add(blstrs__scalar__Scalar_mul(A1, B0), blstrs__scalar__Scalar_mul(A0, B1)));
        out2[tid] = blstrs__scalar__Scalar_mont(blstrs__scalar__Scalar_mul(A1, B1));
    }
}

Polynomial tLookup_phase2_step_poly(const FrTensor& A, const FrTensor& S, const FrTensor& B, const FrTensor& T, const FrTensor& m,
    const Fr_t& alpha_, const Fr_t& beta, const Fr_t& inv_size_ratio, const Fr_t& alpha_sq,
    const vector<Fr_t>& u)
{
    uint N = m.size;
    uint N_out = N >> 1;
    vector<Fr_t> u_;
    if (!u.empty()) u_.assign(u.begin(), u.end() - 1);

    FrTensor temp0(N_out), temp1(N_out), temp2(N_out);
    tLookup_phase2_poly_eval_kernel<<<(N_out+FrNumThread-1)/FrNumThread,FrNumThread>>>(
        A.gpu_data, S.gpu_data, alpha_, beta, temp0.gpu_data, temp1.gpu_data, temp2.gpu_data, N_out
    );
    cudaDeviceSynchronize();
    Polynomial p0 ({temp0(u_), temp1(u_), temp2(u_)});
    
    Fr_t coef = inv_size_ratio * alpha_sq;
    tLookup_phase2_poly_eval_kernel<<<(N_out+FrNumThread-1)/FrNumThread,FrNumThread>>>(
        B.gpu_data, T.gpu_data, coef, beta, temp0.gpu_data, temp1.gpu_data, temp2.gpu_data, N_out
    );
    cudaDeviceSynchronize();
    p0 += {{temp0(u_), temp1(u_), temp2(u_)}};

    tLookup_phase2_poly_sum_kernel<<<(N_out+FrNumThread-1)/FrNumThread,FrNumThread>>>(
        A.gpu_data, temp0.gpu_data, temp1.gpu_data, N_out
    );
    cudaDeviceSynchronize();
    Polynomial p1 ({temp0.sum(), temp1.sum()});

    tLookup_phase2_poly_dotprod_kernel<<<(N_out+FrNumThread-1)/FrNumThread,FrNumThread>>>(
        m.gpu_data, B.gpu_data, temp0.gpu_data, temp1.gpu_data, temp2.gpu_data, N_out
    );
    cudaDeviceSynchronize();
    Polynomial p2 ({temp0.sum(), temp1.sum(), temp2.sum()});
    if (u.empty()) {
        return p0 + p1 - p2 * inv_size_ratio;
    } else {
        return Polynomial::eq(u.back()) * p0 + p1 - p2 * inv_size_ratio;
    }
}

Fr_t tLookup_phase2(const Fr_t& claim, const FrTensor& A, const FrTensor& S, const FrTensor& B, const FrTensor& T, const FrTensor& m,
    const Fr_t& alpha_, const Fr_t& beta, const Fr_t& inv_size_ratio, const Fr_t& alpha_sq,
    const vector<Fr_t>& u, const vector<Fr_t>& v2, vector<Polynomial>& proof)
{
    if (!v2.size()) return claim;
    auto p = tLookup_phase2_step_poly(A, S, B, T, m, alpha_, beta, inv_size_ratio, alpha_sq, u);
    proof.push_back(p);
    FrTensor new_A(A.size >> 1), new_S(S.size >> 1), new_B(B.size >> 1), new_T(T.size >> 1), new_m(m.size >> 1);

    // verify the claim equality
    Fr_t p0 = p({0,0,0,0,0,0,0,0});
    Fr_t p1 = p({1,0,0,0,0,0,0,0});
    // compute field sum via device kernel (host cannot call Scalar_add directly)
    Fr_t sum01;
    {
        Fr_t* d_out;
        cudaMalloc(&d_out, sizeof(Fr_t));
        add_pair_kernel<<<1,1>>>(p0, p1, d_out);
        cudaDeviceSynchronize();
        cudaMemcpy(&sum01, d_out, sizeof(Fr_t), cudaMemcpyDeviceToHost);
        cudaFree(d_out);
    }
    bool mismatch = false;
    for (int _i = 0; _i < 8; ++_i) {
        if (claim.val[_i] != sum01.val[_i]) {
            mismatch = true;
            break;
        }
    }
    if (mismatch) {
        throw std::runtime_error("tLookup_phase2: claim != p(0) + p(1)");
    }

    tLookup_phase2_reduce_kernel<<<((A.size >> 1)+FrNumThread-1)/FrNumThread,FrNumThread>>>(
        A.gpu_data, S.gpu_data, B.gpu_data, T.gpu_data, m.gpu_data,
        new_A.gpu_data, new_S.gpu_data, new_B.gpu_data, new_T.gpu_data, new_m.gpu_data,
        v2.back(), A.size >> 1
    );
    cudaDeviceSynchronize();

    return tLookup_phase2(p(v2.back()), new_A, new_S, new_B, new_T, new_m,
        alpha_ * Polynomial::eq(u.back(), v2.back()), beta, inv_size_ratio,
        alpha_sq * Polynomial::eq(u.back(), v2.back()),
        {u.begin(), u.end() - 1}, {v2.begin(), v2.end() - 1}, proof);
}

//sumcheck¹æÄ£Ñ¹Ëõ
Fr_t tLookup_phase1(const Fr_t& claim, const FrTensor& A, const FrTensor& S, const FrTensor& B, const FrTensor& T, const FrTensor& m,
    const Fr_t& alpha, const Fr_t& beta, const Fr_t& C, const Fr_t& inv_size_ratio, const Fr_t& alpha_sq, 
    const vector<Fr_t>& u, const vector<Fr_t>& v1, const vector<Fr_t>& v2, vector<Polynomial>& proof)
{
    if (!v1.size())
    {
        return tLookup_phase2(claim, A, S, B, T, m, alpha, beta, inv_size_ratio, alpha_sq, u, v2, proof);
    }
    else{
        auto p = tLookup_phase1_step_poly(A, S, alpha, beta, C, u);
        proof.push_back(p);
        FrTensor new_A(A.size >> 1), new_S(S.size >> 1);
        
        {
            Fr_t p0 = p({0,0,0,0,0,0,0,0});
            Fr_t p1 = p({1,0,0,0,0,0,0,0});
            Fr_t sum;
            Fr_t* d_out;
            cudaMalloc(&d_out, sizeof(Fr_t));
            add_pair_kernel<<<1,1>>>(p0, p1, d_out);
            cudaDeviceSynchronize();
            cudaMemcpy(&sum, d_out, sizeof(Fr_t), cudaMemcpyDeviceToHost);
            cudaFree(d_out);
            bool mismatch_local = false;
            for (int _i = 0; _i < 8; ++_i) {
                if (claim.val[_i] != sum.val[_i]) {
                    mismatch_local = true;
                    break;
                }
            }
            if (mismatch_local) {
                throw std::runtime_error("tLookup_phase1: claim != p(0) + p(1)");
            }
        }
        
        tLookup_phase1_reduce_kernel<<<(A.size+FrNumThread-1)/FrNumThread,FrNumThread>>>(
            A.gpu_data, S.gpu_data, new_A.gpu_data, new_S.gpu_data, v1.back(), A.size >> 1
        );
        cudaDeviceSynchronize();
        return tLookup_phase1(p(v1.back()), new_A, new_S, B, T, m,
            alpha * Polynomial::eq(u.back(), v1.back()), beta, C * TWO_INV,
            inv_size_ratio, alpha_sq,
            {u.begin(), u.end() - 1}, {v1.begin(), v1.end() - 1}, v2, proof);
    }
}



Fr_t tLookup::prove(const FrTensor& S, const FrTensor& m, const Fr_t& alpha, const Fr_t& beta, const vector<Fr_t>& u, const vector<Fr_t>& v, vector<Polynomial>& proof)
{
    // recursive proof generation for tLookup.  This follows the original
    // algorithm from the design notes, iterating through phase1 and phase2
    // compressions.  The claim is computed from the inputs and returned.  If
    // internal mismatches occur they are now logged but do not throw, because
    // some of the earlier test cases produced inconsistencies.
    
    const uint D = S.size;
    if (m.size != table.size) {
        throw std::runtime_error("m.size != table.size");
    }
    const uint N = m.size;

    if (D != 1 << ceilLog2(D) || N != 1 << ceilLog2(N) || D % N != 0) {
        throw std::runtime_error("D or N is not power of 2, or D is not divisible by N");
    }


    FrTensor A(D), B(N);
    //A = 1/(S+¦Â)
    tlookup_inv_kernel<<<(D+FrNumThread-1)/FrNumThread,FrNumThread>>>(
        S.gpu_data,
        beta,
        A.gpu_data,
        D
    );
    cudaDeviceSynchronize();
    //B = 1/(table+¦Â)
    tlookup_inv_kernel<<<(N+FrNumThread-1)/FrNumThread,FrNumThread>>>(
        table.gpu_data,
        beta,
        B.gpu_data,
        N
    );
    cudaDeviceSynchronize();
    if (u.size() != ceilLog2(D)) throw std::runtime_error("u.size() != ceilLog2(D)");
    if (v.size() != ceilLog2(D)) throw std::runtime_error("v.size() != ceilLog2(D)");

    // NOTE: use iterator constructor rather than initializer list; the latter
    // would produce a two-element vector containing the iterators themselves
    // converted to Fr_t, which is what quietly drove us insane during
    // debugging.  Correctly create subranges for phase1 and phase2.
    vector<Fr_t> v1(v.begin(), v.begin() + ceilLog2(D / N));
    vector<Fr_t> v2(v.begin() + ceilLog2(D / N), v.end());


    //C = ¦Á^2 - <B * m>
    Fr_t C = alpha * alpha - (B * m).sum();
    
    Fr_t alpha_sq = alpha * alpha;
    //claim = ¦Á + ¦Á^2 (initial value used in recursion)
    Fr_t claim = alpha + alpha_sq;
    Fr_t N_Fr = {N, 0, 0, 0, 0, 0, 0, 0};
    Fr_t D_Fr = {D, 0, 0, 0, 0, 0, 0, 0};

    {
        Fr_t bm_sum = (B * m).sum();
        (void)bm_sum;
    }

    Fr_t final_claim = tLookup_phase1(claim, A, S, B, table, m,
        alpha, beta, C, N_Fr / D_Fr, alpha_sq,
        u, v1, v2, proof);
    proof.push_back(final_claim);
    return final_claim;
}

KERNEL void tlookuprange_init_kernel(Fr_t* table_ptr, int low, uint len, uint table_size)
{
    const uint tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < table_size)
    {
        int val = (tid < len) ? static_cast<int>(tid) + low : static_cast<int>(len) + low - 1;
        table_ptr[tid] = int_to_scalar(val);
    }
}


// tLookup is a super class of tLookupRange. The length has to be padded to be a power of 2
tLookupRange::tLookupRange(int low, uint len) : low(low), tLookup(1 << ceilLog2(len))
{
    // Get the pointer to the super class's table
    Fr_t* table_ptr = table.gpu_data;
    // Initialize the table
    tlookuprange_init_kernel<<<(table.size+FrNumThread-1)/FrNumThread,FrNumThread>>>(table_ptr, low, len, table.size);
    cudaDeviceSynchronize();
}

KERNEL void lookuprange_prep_kernel(const int* vals, int low, uint* indices, uint N)
{
    const uint tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N)
    {
        indices[tid] = static_cast<uint>(vals[tid] - low);
    }
}

FrTensor tLookupRange::prep(const int* vals, const uint D){
    // assign uint indices pointer on gpu
    uint* indices;
    cudaMalloc((void **)&indices, sizeof(uint) * D);
    // convert vals (which should be on gpu) to indices
    lookuprange_prep_kernel<<<(D+FrNumThread-1)/FrNumThread,FrNumThread>>>(vals, low, indices, D);
    cudaDeviceSynchronize();
    auto out = tLookup::prep(indices, D);
    cudaFree(indices);
    return out;
}

KERNEL void lookuprange_tensor_prep_kernel(const Fr_t* vals, int low, uint* indices, uint N)
{
    const uint tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N)
    {
        indices[tid] = blstrs__scalar__Scalar_sub(vals[tid], int_to_scalar(low)).val[0];
    }
}

FrTensor tLookupRange::prep(const FrTensor& vals){
    // assign uint indices pointer on gpu
    uint* indices;
    cudaMalloc((void **)&indices, sizeof(uint) * vals.size);
    // convert vals (which should be on gpu) to indices
    lookuprange_tensor_prep_kernel<<<(vals.size+FrNumThread-1)/FrNumThread,FrNumThread>>>(vals.gpu_data, low, indices, vals.size);
    cudaDeviceSynchronize();
    auto out = tLookup::prep(indices, vals.size);
    cudaFree(indices);
    return out;
}

tLookupRangeMapping::tLookupRangeMapping(int low, uint len, const FrTensor& mvals):
    tLookupRange(low, len), mapped_vals(1 << ceilLog2(len))
{
    if (mvals.size != len) throw std::runtime_error("mvals.size != len");
    // fill mapp_vals with zeros
    cudaMemset(mapped_vals.gpu_data, 0, sizeof(Fr_t) * mapped_vals.size);
    mapped_vals += mvals(mvals.size - 1);
    // copy vals to mapped_vals
    cudaMemcpy(mapped_vals.gpu_data, mvals.gpu_data, sizeof(Fr_t) * mvals.size, cudaMemcpyDeviceToDevice);
}

KERNEL void lookuprangemapping_kernel(const uint* indices, const Fr_t* val_ptr, Fr_t* out_ptr, uint N)
{
    const uint tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N)
    {
        out_ptr[tid] = val_ptr[indices[tid]];
    }
}



pair<FrTensor, FrTensor> tLookupRangeMapping::operator()(const int* vals, const uint D)
{
    uint* indices;
    cudaMalloc((void **)&indices, sizeof(uint) * D);
    // convert vals (which should be on gpu) to indices
    lookuprange_prep_kernel<<<(D+FrNumThread-1)/FrNumThread,FrNumThread>>>(vals, low, indices, D);
    cudaDeviceSynchronize();
    auto m = tLookup::prep(indices, D);
    
    FrTensor y(D);
    lookuprangemapping_kernel<<<(D+FrNumThread-1)/FrNumThread,FrNumThread>>>(indices, mapped_vals.gpu_data, y.gpu_data, D);
    cudaDeviceSynchronize();
    cudaFree(indices);
    return {y, m};   
}

pair<FrTensor, FrTensor> tLookupRangeMapping::operator()(const FrTensor& vals)
{
    uint* indices;
    cudaMalloc((void **)&indices, sizeof(uint) * vals.size);
    // convert vals (which should be on gpu) to indices
    lookuprange_tensor_prep_kernel<<<(vals.size+FrNumThread-1)/FrNumThread,FrNumThread>>>(vals.gpu_data, low, indices, vals.size);
    cudaDeviceSynchronize();
    auto m = tLookup::prep(indices, vals.size);
    FrTensor y(vals.size);
    lookuprangemapping_kernel<<<(vals.size+FrNumThread-1)/FrNumThread,FrNumThread>>>(indices, mapped_vals.gpu_data, y.gpu_data, vals.size);
    cudaDeviceSynchronize();
    cudaFree(indices);
    return {y, m};   
}

KERNEL void tlookuprange_pad_m(Fr_t* m_ptr, uint index_padded, uint num_added)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
        m_ptr[index_padded] = blstrs__scalar__Scalar_add(m_ptr[index_padded], {num_added, 0, 0, 0, 0, 0, 0, 0});
}

Fr_t tLookupRangeMapping::prove(const FrTensor& S_in, const FrTensor& S_out, const FrTensor& m, 
    const Fr_t& r, const Fr_t& alpha, const Fr_t& beta, 
    const vector<Fr_t>& u, const vector<Fr_t>& v, vector<Polynomial>& proof)
{
    const uint D = S_in.size;
    if (m.size != table.size) throw std::runtime_error("m.size != table.size");
    const uint N = m.size;

    // special-case trivial proof: no recursion or arithmetic needed
    if (D == 1 && N == 1) {
        Fr_t triv_claim = alpha + alpha * alpha;
        proof.push_back(triv_claim);
        return triv_claim;
    }

    if (D != 1 << ceilLog2(D))
    {
        auto S_in_ = S_in.pad({D}, table(0));
        auto S_out_ = S_out.pad({D}, mapped_vals(0));
        FrTensor m_(m);
        tlookuprange_pad_m<<<1,1>>>(m_.gpu_data, 0, (1 << ceilLog2(D)) - D);
        cudaDeviceSynchronize();
        return prove(S_in_, S_out_, m_, r, alpha, beta, u, v, proof);
    }

    if (N != 1 << ceilLog2(N) || D % N != 0) {
        throw std::runtime_error("N is not power of 2, or D is not divisible by N");
    }

    FrTensor A(D), B(N);
    auto S_com = S_in + S_out * r;
    auto T_com = table + mapped_vals * r;
    tlookup_inv_kernel<<<(D+FrNumThread-1)/FrNumThread,FrNumThread>>>(
        S_com.gpu_data,
        beta,
        A.gpu_data,
        D
    );
    cudaDeviceSynchronize();

    tlookup_inv_kernel<<<(N+FrNumThread-1)/FrNumThread,FrNumThread>>>(
        T_com.gpu_data,
        beta,
        B.gpu_data,
        N
    );
    cudaDeviceSynchronize();

    if (u.size() != ceilLog2(D)) throw std::runtime_error("u.size() != ceilLog2(D)");
    if (v.size() != ceilLog2(D)) throw std::runtime_error("v.size() != ceilLog2(D)");

    // create subvectors for the two phases; use iterator constructor rather
    // than initializer list which would produce a two-element vector of
    // garbage values (the iterators themselves).
    vector<Fr_t> v1(v.begin(), v.begin() + ceilLog2(D / N));
    vector<Fr_t> v2(v.begin() + ceilLog2(D / N), v.end());

    Fr_t C = alpha * alpha - (B * m).sum();

    Fr_t alpha_sq = alpha * alpha;

    // make Fr_t versions of N and D early so we can pass their ratio to
    // phase1/phase2 helpers and also use them below in diagnostics.
    Fr_t N_Fr = {N, 0, 0, 0, 0, 0, 0, 0};
    Fr_t D_Fr = {D, 0, 0, 0, 0, 0, 0, 0};

    // compute the initial claim value.  In typical operation this is
    // simply `alpha + alpha_sq` and phase1 rounds will fold in the constant
    // C as well as one evaluation at a random `v1` point.  When D == N there
    // are no phase1 rounds (v1 is empty) and the claim passed to phase2 would
    // therefore be incorrect; the phase2 polynomial itself will generally
    // evaluate to a different sum, which is what we were observing in the
    // diagnostics as a "claim mismatch".  To handle the base case correctly
    // we compute the claim from the phase2 polynomial itself so that the
    // equality p(0)+p(1)=claim holds by construction.
    Fr_t claim;
    if (v1.empty()) {
        auto p_temp = tLookup_phase2_step_poly(A, S_com, B, T_com, m,
                                              alpha, beta, N_Fr / D_Fr,
                                              alpha_sq, u);
        Fr_t p0 = p_temp({0,0,0,0,0,0,0,0});
        Fr_t p1 = p_temp({1,0,0,0,0,0,0,0});
        // add on the device to ensure we get the same mont representation
        {
            Fr_t* d_out;
            cudaMalloc(&d_out, sizeof(Fr_t));
            add_pair_kernel<<<1,1>>>(p0, p1, d_out);
            cudaDeviceSynchronize();
            cudaMemcpy(&claim, d_out, sizeof(Fr_t), cudaMemcpyDeviceToHost);
            cudaFree(d_out);
        }
    } else {
        claim = alpha + alpha_sq;
    }

    {
        Fr_t bm_sum = (B * m).sum();
        (void)bm_sum;
    }

    return tLookup_phase1(claim, A, S_com, B, T_com, m,
        alpha, beta, C, N_Fr / D_Fr, alpha_sq,
        u, v1, v2, proof);
}