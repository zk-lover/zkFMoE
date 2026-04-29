#include <iostream>
#include "tlookup.cuh"
#include "fr-tensor.cuh"

int main() {
    const int N = 8;
    FrTensor values(N), table(N);
    FrTensor out(N);
    // prepare host arrays and copy to device
    std::vector<Fr_t> hVals(N), hTab(N);
    for (int i = 0; i < N; ++i) {
        hVals[i] = {static_cast<uint>(i),0,0,0,0,0,0,0};
        hTab[i] = {static_cast<uint>(i*10),0,0,0,0,0,0,0};
    }
    cudaMemcpy(values.gpu_data, hVals.data(), sizeof(Fr_t) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(table.gpu_data, hTab.data(), sizeof(Fr_t) * N, cudaMemcpyHostToDevice);
    std::cout << "=== tlookup max kernel ===\n";
    tlookup_max_kernel<<<(N+FrNumThread-1)/FrNumThread,FrNumThread>>>(values.gpu_data, table.gpu_data, out.gpu_data, N);
    cudaDeviceSynchronize();
    {
        Fr_t host[N];
        cudaMemcpy(host, out.gpu_data, sizeof(host), cudaMemcpyDeviceToHost);
        std::cout << "lookup max result: ";
        for (int i=0;i<N;i++) std::cout<<host[i].val[0]<<",";
        std::cout<<"\n";
    }

    std::cout << "=== tlookup exp kernel ===\n";
    // modify table or values to differentiate
    tlookup_exp_kernel<<<(N+FrNumThread-1)/FrNumThread,FrNumThread>>>(values.gpu_data, table.gpu_data, out.gpu_data, N);
    cudaDeviceSynchronize();
    {
        Fr_t host[N];
        cudaMemcpy(host, out.gpu_data, sizeof(host), cudaMemcpyDeviceToHost);
        std::cout << "lookup exp result: ";
        for (int i=0;i<N;i++) std::cout<<host[i].val[0]<<",";
        std::cout<<"\n";
    }

    std::cout << "=== tlookup recursive kernel ===\n";
    // prepare a prev_state for demonstration
    FrTensor prev(N);
    std::vector<Fr_t> hPrev(N);
    for (int i=0;i<N;i++) hPrev[i] = {static_cast<uint>(i+100),0,0,0,0,0,0,0};
    cudaMemcpy(prev.gpu_data, hPrev.data(), sizeof(Fr_t) * N, cudaMemcpyHostToDevice);
    tlookup_recursive_kernel<<<(N+FrNumThread-1)/FrNumThread,FrNumThread>>>(values.gpu_data, prev.gpu_data, out.gpu_data, N);
    cudaDeviceSynchronize();
    {
        Fr_t host[N];
        cudaMemcpy(host, out.gpu_data, sizeof(host), cudaMemcpyDeviceToHost);
        std::cout << "lookup recursive result: ";
        for (int i=0;i<N;i++) std::cout<<host[i].val[0]<<",";
        std::cout<<"\n";
    }

    // simple proof demonstration using tLookupRangeMapping
    std::cout << "=== tLookupRangeMapping proof (N=8) ===\n";
    // create some sample values on host
    int hvals[N];
    for (int i = 0; i < N; ++i) hvals[i] = i;
    int* d_vals;
    cudaMalloc(&d_vals, sizeof(int) * N);
    cudaMemcpy(d_vals, hvals, sizeof(int) * N, cudaMemcpyHostToDevice);

    // create mapped_vals tensor with same contents as table
    FrTensor mvals(N);
    cudaMemcpy(mvals.gpu_data, hTab.data(), sizeof(Fr_t) * N, cudaMemcpyHostToDevice);

    tLookupRangeMapping rangeMap(0, N, mvals);
    auto pair = rangeMap(d_vals, N);
    FrTensor y = pair.first;
    FrTensor m = pair.second;

    // prepare parameters for proof
    Fr_t r = {1,0,0,0,0,0,0,0};
    Fr_t alpha = {2,0,0,0,0,0,0,0};
    Fr_t beta = {3,0,0,0,0,0,0,0};
    std::vector<Fr_t> u(ceilLog2(N));
    std::vector<Fr_t> v(ceilLog2(N));
    for (size_t i = 0; i < u.size(); ++i) {
        u[i] = {static_cast<uint>(i+1),0,0,0,0,0,0,0};
        v[i] = {static_cast<uint>(i+5),0,0,0,0,0,0,0};
    }
    std::vector<Polynomial> proof;
    try {
        Fr_t res = rangeMap.prove(y, y, m, r, alpha, beta, u, v, proof);
        uint64_t limb0 = res.val[0];
        std::cout << "prove returned limb0=" << limb0 << "\n";
    } catch (const std::exception &e) {
        std::cout << "proof threw exception: " << e.what() << "\n";
    }
    cudaFree(d_vals);

    // trivial proof case with N=1 to ensure success
    {
        const int M = 1;
        std::cout << "=== tLookupRangeMapping proof (N=1) ===\n";
        int hvals2[M] = {0};
        int* d_vals2;
        cudaMalloc(&d_vals2, sizeof(int) * M);
        cudaMemcpy(d_vals2, hvals2, sizeof(int) * M, cudaMemcpyHostToDevice);
        FrTensor mvals2(M);
        Fr_t tmp = {5,0,0,0,0,0,0,0};
        cudaMemcpy(mvals2.gpu_data, &tmp, sizeof(Fr_t) * M, cudaMemcpyHostToDevice);
        tLookupRangeMapping rangeMap2(0, M, mvals2);
        auto pair2 = rangeMap2(d_vals2, M);
        FrTensor y2 = pair2.first;
        FrTensor m2 = pair2.second;
        Fr_t r2 = {0,0,0,0,0,0,0,0};
        Fr_t alpha2 = {0,0,0,0,0,0,0,0};
        Fr_t beta2 = {0,0,0,0,0,0,0,0};
        std::vector<Fr_t> u2(ceilLog2(M));
        std::vector<Fr_t> v2(ceilLog2(M));
        std::vector<Polynomial> proof2;
        try {
            Fr_t res2 = rangeMap2.prove(y2, y2, m2, r2, alpha2, beta2, u2, v2, proof2);
            std::cout << "small proof returned limb0=" << res2.val[0] << "\n";
        } catch (const std::exception &e) {
            std::cout << "small proof threw exception: " << e.what() << "\n";
        }
        cudaFree(d_vals2);
    }

    return 0;
}
