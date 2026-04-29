#include <iostream>
#include "flashmla.cuh"
#include "fr-tensor.cuh"

// forward declaration of conversion kernel defined in fr-tensor.cu
KERNEL void scalar_to_int_kernel(const Fr_t* scalar_ptr, int* int_ptr, uint n);



// no CPU library support available, compute expected value by hand below

int main() {
    std::cout << "flashmla test start\n";
    int m=4, n=4, d=3;
    std::cout << "about to allocate Q\n";
    FrTensor Q(m*d);
    std::cout << "allocated Q\n";
    cudaError_t err = cudaGetLastError(); if (err != cudaSuccess) std::cout << "cuda error after Q: " << cudaGetErrorString(err) << "\n";
    FrTensor K(n*d);
    std::cout << "allocated K\n";
    err = cudaGetLastError(); if (err != cudaSuccess) std::cout << "cuda error after K: " << cudaGetErrorString(err) << "\n";
    FrTensor out(m*n);
    std::cout << "allocated out\n";
    err = cudaGetLastError(); if (err != cudaSuccess) std::cout << "cuda error after out: " << cudaGetErrorString(err) << "\n";
    FrTensor expected(m*n);
    std::cout << "allocated expected\n";
    err = cudaGetLastError(); if (err != cudaSuccess) std::cout << "cuda error after expected: " << cudaGetErrorString(err) << "\n";
    // fill with small integers on host then copy to device
    std::cout << "about to fill tensors\n";
    Fr_t one = {1,0,0,0,0,0,0,0};
    Fr_t two = {2,0,0,0,0,0,0,0};
    std::vector<Fr_t> hQ_init(m*d, one);
    std::vector<Fr_t> hK_init(n*d, two);
    cudaMemcpy(Q.gpu_data, hQ_init.data(), sizeof(Fr_t) * (m*d), cudaMemcpyHostToDevice);
    std::cout << "filled Q\n";
    cudaMemcpy(K.gpu_data, hK_init.data(), sizeof(Fr_t) * (n*d), cudaMemcpyHostToDevice);
    std::cout << "filled K\n";
    // inputs will be converted to mont form inside kernel

    FlashMLAConfig cfg{ /*Bq=*/2u, /*Bk=*/2u, /*Bd=*/(unsigned)d };
    std::cout << "constructed cfg\n";
    // start with empty tensors and let helper allocate correct size
    FrTensor block_max(0);
    std::cout << "constructed block_max\n";
    FrTensor block_sum(0);
    std::cout << "constructed block_sum\n";
    std::cout << "about to call allocate_flash_blocks\n";
    allocate_flash_blocks(block_max, block_sum, m, n, cfg);
    std::cout << "returned from allocate_flash_blocks\n";
    std::cout << "launching flashmla kernel\n";
    flashmla_kernel(Q, K, out, cfg, block_max, block_sum);
    err = cudaGetLastError();
    if (err != cudaSuccess) std::cout << "kernel launch error: " << cudaGetErrorString(err) << "\n";
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) std::cout << "kernel sync error: " << cudaGetErrorString(err) << "\n";

    // verify Q and K on host
    std::vector<Fr_t> hQ_host(m*d), hK_host(n*d), hOut(m*n);
    cudaMemcpy(hQ_host.data(), Q.gpu_data, sizeof(Fr_t) * m * d, cudaMemcpyDeviceToHost);
    cudaMemcpy(hK_host.data(), K.gpu_data, sizeof(Fr_t) * n * d, cudaMemcpyDeviceToHost);
    cudaMemcpy(hOut.data(), out.gpu_data, sizeof(Fr_t) * m * n, cudaMemcpyDeviceToHost);
    std::cout << "host raw limbs Q[0]=" << hQ_host[0].val[0] << " K[0]=" << hK_host[0].val[0] << "\n";

    // decode entire output and block arrays to ints
    int total = m * n;
    int *d_int;
    cudaMalloc(&d_int, sizeof(int) * total);
    scalar_to_int_kernel<<<(total+255)/256,256>>>(out.gpu_data, d_int, total);
    std::vector<int> host_int_out(total);
    cudaMemcpy(host_int_out.data(), d_int, sizeof(int) * total, cudaMemcpyDeviceToHost);

    uint blk_count = ((m + 2 - 1)/2) * ((n + 2 - 1)/2);
    int *d_blk;
    cudaMalloc(&d_blk, sizeof(int) * blk_count);
    scalar_to_int_kernel<<<(blk_count+255)/256,256>>>(block_max.gpu_data, d_blk, blk_count);
    std::vector<int> host_blk_max(blk_count);
    cudaMemcpy(host_blk_max.data(), d_blk, sizeof(int) * blk_count, cudaMemcpyDeviceToHost);
    scalar_to_int_kernel<<<(blk_count+255)/256,256>>>(block_sum.gpu_data, d_blk, blk_count);
    std::vector<int> host_blk_sum(blk_count);
    cudaMemcpy(host_blk_sum.data(), d_blk, sizeof(int) * blk_count, cudaMemcpyDeviceToHost);
    cudaFree(d_int);
    cudaFree(d_blk);

    // verify values
    int expected_int = 2 * d;
    bool ok = true;
    for (int i = 0; i < total; ++i) {
        if (host_int_out[i] != expected_int) ok = false;
    }
    std::cout << "all outputs correct? " << ok << "\n";
    std::cout << "block max values:";
    for (int i = 0; i < blk_count; ++i) std::cout << " " << host_blk_max[i];
    std::cout << "\nblock sum values:";
    for (int i = 0; i < blk_count; ++i) std::cout << " " << host_blk_sum[i];
    std::cout << "\n";

    // print one example
    std::cout << "first element (raw limb) out=" << hOut[0].val[0] << " expected=" << expected_int << "\n";
    std::cout << "first element (decoded) out=" << host_int_out[0] << " expected_int=" << expected_int << "\n";
    return 0;
}
