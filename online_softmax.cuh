#pragma once

#include "fr-tensor.cuh"
#include "polynomial.cuh"

// context carrying lookup tables and internal state for online softmax
struct OnlineSoftmaxCtx {
    FrTensor max_table;      // for max lookup (indexed by raw value)
    FrTensor exp_table;      // for exp lookup
    FrTensor state_max;      // running maximum per row/position
    FrTensor state_sum;      // running sum per row/position
};

// kernel prototypes used by the streaming algorithm
KERNEL void online_softmax_step_kernel(const Fr_t* X, Fr_t* Y,
                                       const Fr_t* exp_table,
                                       Fr_t* state_max, Fr_t* state_sum,
                                       uint rows, uint cols);
KERNEL void online_softmax_finalize_kernel(Fr_t* Y, const Fr_t* state_sum,
                                           uint rows, uint cols);

// helper for tests: convert array elements to montgomery form
KERNEL void to_mont_kernel(Fr_t* arr, uint n);
// helper for tests: convert from montgomery representation back to standard
KERNEL void unmont_kernel(Fr_t* arr, uint n);

// interface for compute/prove
FrTensor online_softmax_compute(const FrTensor &X, OnlineSoftmaxCtx &ctx,
                                FrTensor &shift, FrTensor &X_shifted,
                                vector<FrTensor> &aux_states);

Fr_t online_softmax_prove(const FrTensor &Y, const FrTensor &X,
                          OnlineSoftmaxCtx &ctx,
                          const FrTensor &shift, const FrTensor &X_shifted,
                          const vector<FrTensor> &aux_states,
                          const vector<Fr_t> &u_Y, const vector<Fr_t> &v_Y,
                          vector<Polynomial> &proof);
