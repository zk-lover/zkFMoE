#include <iostream>
#include "online_softmax.cuh"
#include "fr-tensor.cuh"

// (no need for scalar->int helper; we'll decode on host)


int main() {
    std::cout << "test_online: start" << std::endl;
    // small 2x2 matrix
    uint rows = 2;
    uint cols = 2;
    std::cout << "allocating X" << std::endl;
    FrTensor X(rows * cols);
    // initialize X with simple increasing values
    Fr_t vals[4] = {{1,0,0,0,0,0,0,0}, {2,0,0,0,0,0,0,0},
                    {3,0,0,0,0,0,0,0}, {4,0,0,0,0,0,0,0}};
    cudaMemcpy(X.gpu_data, vals, sizeof(vals), cudaMemcpyHostToDevice);

    FrTensor shift(rows); // not used by our stub but required
    FrTensor X_shifted(rows * cols);
    vector<FrTensor> aux_states;
    std::cout << "creating context" << std::endl;
    OnlineSoftmaxCtx ctx{FrTensor(0), FrTensor(0), FrTensor(0), FrTensor(0)};
    // prepare a small exponent table (identity mapping for simplicity)
    int tableSize = 16;
    ctx.exp_table = FrTensor(tableSize);
    std::vector<Fr_t> htab(tableSize);
    for (int i = 0; i < tableSize; ++i) htab[i] = {static_cast<uint>(i),0,0,0,0,0,0,0};
    cudaMemcpy(ctx.exp_table.gpu_data, htab.data(), sizeof(Fr_t) * tableSize, cudaMemcpyHostToDevice);
    std::cout << "context created with exp_table" << std::endl;

    std::cout << "calling compute" << std::endl;
    FrTensor Y = online_softmax_compute(X, ctx, shift, X_shifted, aux_states);
    std::cout << "returned from compute" << std::endl;
    // unmont results before copying back
    uint N = rows * cols;
    uint threads = FrNumThread;
    uint blocks = (N + threads - 1) / threads;
    unmont_kernel<<<blocks, threads>>>(Y.gpu_data, N);
    cudaDeviceSynchronize();

    // copy back Y to host and print limbs
    Fr_t hostY[4];
    cudaMemcpy(hostY, Y.gpu_data, sizeof(hostY), cudaMemcpyDeviceToHost);
    std::cout << "Y raw results (limb0,limb1):\n";
    for (int i = 0; i < 4; ++i) {
        uint64_t l0 = hostY[i].val[0];
        uint64_t l1 = hostY[i].val[1];
        std::cout << "(" << l0 << "," << l1 << "),";
    }
    std::cout << std::endl;
    // value estimate using first limb
    std::cout << "Y estimated ints (limb0):\n";
    for (int i = 0; i < 4; ++i) {
        uint64_t l0 = hostY[i].val[0];
        std::cout << l0 << ",";
    }
    std::cout << std::endl;
    // print state_max and state_sum
    Fr_t host_max[2], host_sum[2];
    cudaMemcpy(host_max, ctx.state_max.gpu_data, sizeof(host_max), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_sum, ctx.state_sum.gpu_data, sizeof(host_sum), cudaMemcpyDeviceToHost);
    // unmont state buffers for easier reading
    unmont_kernel<<<(rows+threads-1)/threads,threads>>>(ctx.state_max.gpu_data, rows);
    unmont_kernel<<<(rows+threads-1)/threads,threads>>>(ctx.state_sum.gpu_data, rows);
    cudaDeviceSynchronize();
    cudaMemcpy(host_max, ctx.state_max.gpu_data, sizeof(host_max), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_sum, ctx.state_sum.gpu_data, sizeof(host_sum), cudaMemcpyDeviceToHost);
    std::cout << "state_max:" << host_max[0].val[0] << "," << host_max[1].val[0] << std::endl;
    std::cout << "state_sum:" << host_sum[0].val[0] << "," << host_sum[1].val[0] << std::endl;

    // now call the proof routine using simple randomness
    std::vector<Fr_t> uY(ceilLog2(N));
    std::vector<Fr_t> vY(ceilLog2(N));
    for (size_t i = 0; i < uY.size(); ++i) {
        uY[i] = {static_cast<uint>(i+1),0,0,0,0,0,0,0};
        vY[i] = {static_cast<uint>(i+2),0,0,0,0,0,0,0};
    }
    std::vector<Polynomial> proof;
    try {
        Fr_t proof_res = online_softmax_prove(Y, X, ctx, shift, X_shifted, aux_states, uY, vY, proof);
        std::cout << "online_softmax_prove result limb0=" << proof_res.val[0] << "\n";
    } catch (const std::exception &e) {
        std::cout << "online_softmax_prove threw: " << e.what() << "\n";
    }

    return 0;
}
