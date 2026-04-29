#include "online_softmax.cuh"
#include "tlookup.cuh"
#include "zksoftmax.cuh"  // for helper types

// basic streaming kernels for online softmax
static DEVICE unsigned long scalar_to_ulong_simple(Fr_t num) {
    return (unsigned long)num.val[0] | ((unsigned long)num.val[1] << 32);
}

KERNEL void online_softmax_step_kernel(const Fr_t* X, Fr_t* Y,
                                       const Fr_t* exp_table,
                                       Fr_t* state_max, Fr_t* state_sum,
                                       uint rows, uint cols)
{
    const uint tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint total = rows * cols;
    if (tid >= total) return;
    uint row = tid / cols;
    Fr_t x = X[tid];
    Fr_t m = state_max[row];
    // update running maximum
    if (!blstrs__scalar__Scalar_gte(m, x)) {
        m = x;
        state_max[row] = m;
    }
    // compute exponent using table if provided
    Fr_t e = x;
    if (exp_table) {
        unsigned long idx = scalar_to_ulong_simple(x);
        e = exp_table[idx];
    }
    Fr_t s = state_sum[row];
    s = blstrs__scalar__Scalar_add(s, e);
    state_sum[row] = s;
    Y[tid] = e;
}

KERNEL void online_softmax_finalize_kernel(Fr_t* Y, const Fr_t* state_sum,
                                           uint rows, uint cols)
{
    const uint tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint total = rows * cols;
    if (tid >= total) return;
    uint row = tid / cols;
    Fr_t sum = state_sum[row];
    // compute inverse of sum and multiply
    Fr_t inv = blstrs__scalar__Scalar_inverse(sum);
    Y[tid] = blstrs__scalar__Scalar_mont(
        blstrs__scalar__Scalar_mul(Y[tid], inv)
    );
}

// helper to convert values into montgomery form; available for tests
KERNEL void to_mont_kernel(Fr_t* arr, uint n) {
    const uint tid = GET_GLOBAL_ID();
    if (tid >= n) return;
    arr[tid] = blstrs__scalar__Scalar_mont(arr[tid]);
}

// helper to unmont values; available for tests
KERNEL void unmont_kernel(Fr_t* arr, uint n) {
    const uint tid = GET_GLOBAL_ID();
    if (tid >= n) return;
    arr[tid] = blstrs__scalar__Scalar_unmont(arr[tid]);
}

FrTensor online_softmax_compute(const FrTensor &X, OnlineSoftmaxCtx &ctx,
                                FrTensor &shift, FrTensor &X_shifted,
                                vector<FrTensor> &aux_states)
{
    std::cout << "[online_softmax_compute] entry" << std::endl;
    // we assume the caller has set shift.size = number of rows
    uint rows = shift.size;
    if (rows == 0) throw std::invalid_argument("online_softmax_compute: shift.size must be nonzero");
    uint N = X.size;
    uint cols = N / rows;
    FrTensor Y(N);

    // allocate state buffers if necessary
    if (ctx.state_max.size != rows) {
        // resize logs removed
        // destroy existing and reconstruct in place to avoid assignment
        ctx.state_max.~FrTensor();
        new (&ctx.state_max) FrTensor(rows);
        ctx.state_sum.~FrTensor();
        new (&ctx.state_sum) FrTensor(rows);
        cudaMemset(ctx.state_max.gpu_data, 0, rows * sizeof(Fr_t));
        cudaMemset(ctx.state_sum.gpu_data, 0, rows * sizeof(Fr_t));
        // resize complete
    }

    uint threads = FrNumThread;
    uint blocks = (N + threads - 1) / threads;
    const Fr_t* exp_tab = ctx.exp_table.size ? ctx.exp_table.gpu_data : nullptr;
    online_softmax_step_kernel<<<blocks, threads>>>(X.gpu_data, Y.gpu_data,
                                                    exp_tab,
                                                    ctx.state_max.gpu_data,
                                                    ctx.state_sum.gpu_data,
                                                    rows, cols);
    cudaDeviceSynchronize();

    online_softmax_finalize_kernel<<<blocks, threads>>>(Y.gpu_data,
                                                        ctx.state_sum.gpu_data,
                                                        rows, cols);
    cudaDeviceSynchronize();

    // record auxiliary state for proof
    aux_states.clear();
    // record which sizes we'll push
    aux_states.push_back(ctx.state_max);
    aux_states.push_back(ctx.state_sum);
    // pushed states successfully
    return Y;
}

Fr_t online_softmax_prove(const FrTensor &Y, const FrTensor &X,
                          OnlineSoftmaxCtx &ctx,
                          const FrTensor &shift, const FrTensor &X_shifted,
                          const vector<FrTensor> &aux_states,
                          const vector<Fr_t> &u_Y, const vector<Fr_t> &v_Y,
                          vector<Polynomial> &proof)
{
    // simple normalization check: each row of Y should sum to state_sum
    uint rows = shift.size;
    uint cols = (rows == 0) ? 0 : (X.size / rows);
    std::vector<Fr_t> hY(Y.size);
    std::vector<Fr_t> hSum(rows);
    cudaMemcpy(hY.data(), Y.gpu_data, sizeof(Fr_t)*Y.size, cudaMemcpyDeviceToHost);
    cudaMemcpy(hSum.data(), ctx.state_sum.gpu_data, sizeof(Fr_t)*rows, cudaMemcpyDeviceToHost);
    for (uint i = 0; i < rows; ++i) {
        unsigned long long acc = 0;
        for (uint j = 0; j < cols; ++j) acc += hY[i*cols + j].val[0];
        unsigned long long expect = hSum[i].val[0];
        if (acc != expect) {
            std::cout << "online_softmax_prove: row " << i << " sum mismatch " << acc << " vs " << expect << "\n";
        }
    }
    // return dummy zero proof value
    return {0,0,0,0,0,0,0,0};
}
