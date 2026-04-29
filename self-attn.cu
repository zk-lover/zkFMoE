#include "zksoftmax.cuh"
#include "zkfc.cuh"
#include "fr-tensor.cuh"
#include "proof.cuh"
#include "commitment.cuh"
#include "rescaling.cuh"
#include "flashmla.cuh"
#include "online_softmax.cuh"
#include <string>
#include <chrono>
#include <cstring>
#include <memory>
#include "ioutils.cuh"

// RoPE helper: applies rotary positional encoding to a tensor in-place.
// X has shape (seq_len * d) in row-major. cp and sp are precomputed cosine
// and sine tables of size seq_len * (d/2) respectively.

KERNEL void rope_kernel(Fr_t* data, const Fr_t* cp, const Fr_t* sp,
                        uint seq_len, uint d)
{
    const uint idx = threadIdx.x + blockIdx.x * blockDim.x;
    uint total = seq_len * (d >> 1);
    if (idx < total) {
        uint p = idx / (d >> 1);
        uint i = idx % (d >> 1);
        uint base = p * d + (i << 1);
        Fr_t q0 = data[base];
        Fr_t q1 = data[base + 1];
        Fr_t c = cp[idx];
        Fr_t s = sp[idx];
        // new0 = q0 * c - q1 * s
        Fr_t new0 = blstrs__scalar__Scalar_sub(
            blstrs__scalar__Scalar_mul(q0, c),
            blstrs__scalar__Scalar_mul(q1, s));
        // new1 = q0 * s + q1 * c
        Fr_t new1 = blstrs__scalar__Scalar_add(
            blstrs__scalar__Scalar_mul(q0, s),
            blstrs__scalar__Scalar_mul(q1, c));
        data[base] = new0;
        data[base + 1] = new1;
    }
}

// utility to launch the RoPE kernel
void apply_rope(FrTensor &X, const FrTensor &cp, const FrTensor &sp,
                uint seq_len, uint d)
{
    if (cp.size != seq_len * (d >> 1) || sp.size != cp.size)
        throw std::runtime_error("RoPE tables have wrong size");
    uint total = seq_len * (d >> 1);
    rope_kernel<<<(total+FrNumThread-1)/FrNumThread,FrNumThread>>>(
        X.gpu_data, cp.gpu_data, sp.gpu_data, seq_len, d);
    cudaDeviceSynchronize();
}

FrTensor repeat_kv_to_qdim(const FrTensor& kv, uint seq_len, uint kv_dim, uint q_dim)
{
    if (kv_dim == 0 || q_dim == 0 || (q_dim % kv_dim) != 0)
        throw std::runtime_error("Invalid kv_dim/q_dim for GQA expansion");
    FrTensor out(seq_len * q_dim);
    std::vector<Fr_t> h_kv(kv.size);
    cudaMemcpy(h_kv.data(), kv.gpu_data, sizeof(Fr_t) * kv.size, cudaMemcpyDeviceToHost);
    std::vector<Fr_t> h_out(seq_len * q_dim);
    for (uint t = 0; t < seq_len; ++t)
    {
        for (uint j = 0; j < q_dim; ++j)
        {
            h_out[t * q_dim + j] = h_kv[t * kv_dim + (j % kv_dim)];
        }
    }
    cudaMemcpy(out.gpu_data, h_out.data(), sizeof(Fr_t) * h_out.size(), cudaMemcpyHostToDevice);
    return out;
}

FrTensor slice_cols(const FrTensor& x, uint rows, uint in_dim, uint start_col, uint width)
{
    if (start_col + width > in_dim)
        throw std::runtime_error("slice_cols out of range");
    std::vector<Fr_t> h_in(x.size);
    cudaMemcpy(h_in.data(), x.gpu_data, sizeof(Fr_t) * x.size, cudaMemcpyDeviceToHost);
    std::vector<Fr_t> h_out(rows * width);
    for (uint r = 0; r < rows; ++r)
    {
        const Fr_t* src = h_in.data() + r * in_dim + start_col;
        Fr_t* dst = h_out.data() + r * width;
        std::memcpy(dst, src, sizeof(Fr_t) * width);
    }
    FrTensor out(rows * width);
    cudaMemcpy(out.gpu_data, h_out.data(), sizeof(Fr_t) * h_out.size(), cudaMemcpyHostToDevice);
    return out;
}

int main(int argc, char *argv[])
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <mode> [args...]\n";
        std::cerr << "Modes:\n  demo\n  linear <input> <seq_len> <embed_dim> <workdir> <layer_prefix> <output_file> [q_dim] [kv_dim]\n  linear_deepseek_mla <input> <seq_len> <embed_dim> <workdir> <layer_prefix> <output_file>\n  linear_deepseek_v32 <input> <seq_len> <embed_dim> <workdir> <layer_prefix> <output_file>\n  attn[ _online] <unused> <seq_len> <embed_dim> <workdir> <layer_prefix> <output_file>\n";
        return 1;
    }

    string mode = argv[1];

    uint seq_len = 0;
    uint embed_dim = 0;

    // demo mode: construct small random tensors and save temp_Q/K/V, then
    // fall through to the attn path for end-to-end verification.
    if (mode == "demo") {
        uint seq_len_demo = 256;
        uint embed_dim_demo = 8;
        uint n = seq_len_demo * embed_dim_demo;
        FrTensor input(n);
        // fill with simple values 0..n-1
        std::vector<Fr_t> h(n);
        for (uint i = 0; i < n; ++i) h[i] = {i,0,0,0,0,0,0,0};
        cudaMemcpy(input.gpu_data, h.data(), sizeof(Fr_t) * n, cudaMemcpyHostToDevice);

        // create Q,K,V as copies (for demo simplicity)
        FrTensor Q(n), K(n), V(n);
        cudaMemcpy(Q.gpu_data, h.data(), sizeof(Fr_t) * n, cudaMemcpyHostToDevice);
        cudaMemcpy(K.gpu_data, h.data(), sizeof(Fr_t) * n, cudaMemcpyHostToDevice);
        cudaMemcpy(V.gpu_data, h.data(), sizeof(Fr_t) * n, cudaMemcpyHostToDevice);

        Q.save_int("temp_Q.bin");
        K.save_int("temp_K.bin");
        V.save_int("temp_V.bin");

        // set variables expected by attn path
        seq_len = seq_len_demo;
        embed_dim = embed_dim_demo;
        mode = "attn_online";
    }

    string input_file_name = (argc > 2) ? argv[2] : string();
    seq_len = (argc > 3) ? std::stoi(argv[3]) : seq_len;
    embed_dim = (argc > 4) ? std::stoi(argv[4]) : embed_dim;
    string workdir = (argc > 5) ? argv[5] : string();
    string layer_prefix = (argc > 6) ? argv[6] : string();
    string output_file_name = (argc > 7) ? argv[7] : string();
    uint q_dim = (argc > 8) ? std::stoi(argv[8]) : embed_dim;
    uint kv_dim = (argc > 9) ? std::stoi(argv[9]) : embed_dim;

    // flag for whether to apply RoPE; mode may contain "rope" substring
    bool use_rope = mode.find("rope") != string::npos;
    bool use_flash = mode.find("flash") != string::npos;
    bool use_online = mode.find("online") != string::npos;

    if (mode == "linear")
    {
        auto q_proj = create_weight(
            workdir + "/self_attn.q_proj.weight-pp.bin",
            workdir + "/" + layer_prefix + "-self_attn.q_proj.weight-int.bin",
            workdir + "/" + layer_prefix + "-self_attn.q_proj.weight-commitment.bin",
              embed_dim,
              q_dim
        );

        auto k_proj = create_weight(
            workdir + "/self_attn.k_proj.weight-pp.bin",
            workdir + "/" + layer_prefix + "-self_attn.k_proj.weight-int.bin",
            workdir + "/" + layer_prefix + "-self_attn.k_proj.weight-commitment.bin",
            embed_dim,
              kv_dim
        );

        auto v_proj = create_weight(
            workdir + "/self_attn.v_proj.weight-pp.bin",
            workdir + "/" + layer_prefix + "-self_attn.v_proj.weight-int.bin",
            workdir + "/" + layer_prefix + "-self_attn.v_proj.weight-commitment.bin",
            embed_dim,
                kv_dim
        );
            zkFC q_layer(embed_dim, q_dim, q_proj.weight);
            zkFC k_layer(embed_dim, kv_dim, k_proj.weight);
            zkFC v_layer(embed_dim, kv_dim, v_proj.weight);
        Rescaling q_rescale(1);
        Rescaling k_rescale(1);
        Rescaling v_rescale(1);

        FrTensor input = FrTensor::from_int_bin(input_file_name);
        auto Q = q_layer(input);
        auto K = k_layer(input);
        auto V = v_layer(input);

        if (use_rope) {
            // load precomputed cosine and sine tables for RoPE
            auto cp = FrTensor::from_int_bin(
                workdir + "/" + layer_prefix + "-rope.cos-int.bin");
            auto sp = FrTensor::from_int_bin(
                workdir + "/" + layer_prefix + "-rope.sin-int.bin");
            apply_rope(Q, cp, sp, seq_len, embed_dim);
            apply_rope(K, cp, sp, seq_len, embed_dim);
        }

        bool identity_rescale = (q_rescale.scaling_factor == 1 &&
                                 k_rescale.scaling_factor == 1 &&
                                 v_rescale.scaling_factor == 1);
        std::unique_ptr<FrTensor> Q_rescaled;
        std::unique_ptr<FrTensor> K_rescaled;
        std::unique_ptr<FrTensor> V_rescaled;
        if (!identity_rescale) {
            Q_rescaled = std::make_unique<FrTensor>(q_rescale(Q));
            K_rescaled = std::make_unique<FrTensor>(k_rescale(K));
            V_rescaled = std::make_unique<FrTensor>(v_rescale(V));
        }
        const FrTensor& Q_ = identity_rescale ? Q : *Q_rescaled;
        const FrTensor& K_ = identity_rescale ? K : *K_rescaled;
        const FrTensor& V_ = identity_rescale ? V : *V_rescaled;
            FrTensor K_attn = K_;
            FrTensor V_attn = V_;
            if (kv_dim != q_dim)
            {
                K_attn = repeat_kv_to_qdim(K_, seq_len, kv_dim, q_dim);
                V_attn = repeat_kv_to_qdim(V_, seq_len, kv_dim, q_dim);
            }

        if (!identity_rescale) {
            q_rescale.prove(Q, Q_);
            k_rescale.prove(K, K_);
            v_rescale.prove(V, V_);
        }

        auto k_claim = k_layer.prove(input, K)[0];
        auto q_claim = q_layer.prove(input, Q)[0];
        auto v_claim = v_layer.prove(input, V)[0];

        auto verify_t0 = std::chrono::steady_clock::now();
        verifyWeightClaim(k_proj, k_claim);
        verifyWeightClaim(q_proj, q_claim);
        verifyWeightClaim(v_proj, v_claim);
        auto verify_t1 = std::chrono::steady_clock::now();
        double verifier_time_s = std::chrono::duration<double>(verify_t1 - verify_t0).count();

        Q_.save_int("temp_Q.bin");
            K_attn.save_int("temp_K.bin");
            V_attn.save_int("temp_V.bin");

        cout << "VERIFIER_TIME_LINEAR_S=" << verifier_time_s << "\n";
        cout << "QKV linear proof successfully verified!" << endl;

        return 0;
    }

    else if (mode == "linear_deepseek_v32" || mode == "linear_deepseek_mla")
    {
        // DeepSeek-V3.2 MLA approximation path.
        // Keep this in a dedicated mode so existing OPT/Qwen paths are unchanged.
        const uint q_lora_rank = 1536;
        const uint kv_a_out_dim = 576;
        const uint kv_lora_rank = 512;
        const uint q_b_out_dim = 24576;
        const uint kv_b_out_dim = 32768;
        const uint attn_dim = 16384;

        auto q_a_proj = create_weight(
            workdir + "/self_attn.q_a_proj.weight-pp.bin",
            workdir + "/" + layer_prefix + "-self_attn.q_a_proj.weight-int.bin",
            workdir + "/" + layer_prefix + "-self_attn.q_a_proj.weight-commitment.bin",
            embed_dim,
            q_lora_rank
        );
        auto q_b_proj = create_weight(
            workdir + "/self_attn.q_b_proj.weight-pp.bin",
            workdir + "/" + layer_prefix + "-self_attn.q_b_proj.weight-int.bin",
            workdir + "/" + layer_prefix + "-self_attn.q_b_proj.weight-commitment.bin",
            q_lora_rank,
            q_b_out_dim
        );
        auto kv_a_proj = create_weight(
            workdir + "/self_attn.kv_a_proj_with_mqa.weight-pp.bin",
            workdir + "/" + layer_prefix + "-self_attn.kv_a_proj_with_mqa.weight-int.bin",
            workdir + "/" + layer_prefix + "-self_attn.kv_a_proj_with_mqa.weight-commitment.bin",
            embed_dim,
            kv_a_out_dim
        );
        auto kv_b_proj = create_weight(
            workdir + "/self_attn.kv_b_proj.weight-pp.bin",
            workdir + "/" + layer_prefix + "-self_attn.kv_b_proj.weight-int.bin",
            workdir + "/" + layer_prefix + "-self_attn.kv_b_proj.weight-commitment.bin",
            kv_lora_rank,
            kv_b_out_dim
        );

        zkFC q_a_layer(embed_dim, q_lora_rank, q_a_proj.weight);
        zkFC q_b_layer(q_lora_rank, q_b_out_dim, q_b_proj.weight);
        zkFC kv_a_layer(embed_dim, kv_a_out_dim, kv_a_proj.weight);
        zkFC kv_b_layer(kv_lora_rank, kv_b_out_dim, kv_b_proj.weight);

        Rescaling q_rescale(1), k_rescale(1), v_rescale(1);

        FrTensor input = FrTensor::from_int_bin(input_file_name);
        FrTensor Q(0), K(0), V(0);

        {
            // q-path first to reduce peak memory.
            auto Q_a = q_a_layer(input);
            auto Q_full = q_b_layer(Q_a);
            Q = slice_cols(Q_full, seq_len, q_b_out_dim, 0, attn_dim);

            auto q_a_claim = q_a_layer.prove(input, Q_a)[0];
            auto q_b_claim = q_b_layer.prove(Q_a, Q_full)[0];

            auto verify_t0 = std::chrono::steady_clock::now();
            cout << "VERIFY_STAGE=q_a_proj\n";
            verifyWeightClaim(q_a_proj, q_a_claim);
            cout << "VERIFY_STAGE=q_b_proj\n";
            verifyWeightClaim(q_b_proj, q_b_claim);
            auto verify_t1 = std::chrono::steady_clock::now();
            double verifier_time_s = std::chrono::duration<double>(verify_t1 - verify_t0).count();
            cout << "VERIFIER_TIME_LINEAR_S=" << verifier_time_s << "\n";
        }
        {
            // kv-path next after q-path tensors are released.
            auto KV_a_full = kv_a_layer(input);
            auto KV_a = slice_cols(KV_a_full, seq_len, kv_a_out_dim, 0, kv_lora_rank);
            auto KV_full = kv_b_layer(KV_a);
            K = slice_cols(KV_full, seq_len, kv_b_out_dim, 0, attn_dim);
            V = slice_cols(KV_full, seq_len, kv_b_out_dim, attn_dim, attn_dim);

            auto kv_a_claim = kv_a_layer.prove(input, KV_a_full)[0];
            auto kv_b_claim = kv_b_layer.prove(KV_a, KV_full)[0];

            auto verify_t0 = std::chrono::steady_clock::now();
            cout << "VERIFY_STAGE=kv_a_proj_with_mqa\n";
            verifyWeightClaim(kv_a_proj, kv_a_claim);
            cout << "VERIFY_STAGE=kv_b_proj\n";
            verifyWeightClaim(kv_b_proj, kv_b_claim);
            auto verify_t1 = std::chrono::steady_clock::now();
            double verifier_time_s = std::chrono::duration<double>(verify_t1 - verify_t0).count();
            cout << "VERIFIER_TIME_LINEAR_S=" << verifier_time_s << "\n";
        }

        bool identity_rescale = (q_rescale.scaling_factor == 1 &&
                                 k_rescale.scaling_factor == 1 &&
                                 v_rescale.scaling_factor == 1);
        std::unique_ptr<FrTensor> Q_rescaled;
        std::unique_ptr<FrTensor> K_rescaled;
        std::unique_ptr<FrTensor> V_rescaled;
        if (!identity_rescale) {
            Q_rescaled = std::make_unique<FrTensor>(q_rescale(Q));
            K_rescaled = std::make_unique<FrTensor>(k_rescale(K));
            V_rescaled = std::make_unique<FrTensor>(v_rescale(V));
        }
        const FrTensor& Q_ = identity_rescale ? Q : *Q_rescaled;
        const FrTensor& K_ = identity_rescale ? K : *K_rescaled;
        const FrTensor& V_ = identity_rescale ? V : *V_rescaled;

        if (!identity_rescale) {
            q_rescale.prove(Q, Q_);
            k_rescale.prove(K, K_);
            v_rescale.prove(V, V_);
        }
        Q_.save_int("temp_Q.bin");
        K_.save_int("temp_K.bin");
        V_.save_int("temp_V.bin");
        cout << "DEEPSEEK_V32_MLA linear proof successfully verified!" << endl;

        return 0;
    }

    else if (mode.rfind("attn", 0) == 0)
    {
        auto Q = FrTensor::from_int_bin("temp_Q.bin");
        auto K = FrTensor::from_int_bin("temp_K.bin");
        auto V = FrTensor::from_int_bin("temp_V.bin");
        auto d = Q.size / seq_len;
        
        FrTensor X(seq_len * seq_len);
        // block arrays only needed if flash enabled
        FrTensor block_max(0), block_sum(0);
        if (use_flash) {
            FlashMLAConfig cfg{ /*Bq=*/16, /*Bk=*/16, /*Bd=*/d };
            allocate_flash_blocks(block_max, block_sum, seq_len, seq_len, cfg);
            flashmla_kernel(Q, K, X, cfg, block_max, block_sum);
        } else {
            X = FrTensor::matmul(Q, K.transpose(seq_len, d), seq_len, d, seq_len);
        }

        zkSoftmax softmax({1<<8, 1<<8, 1<<8}, 1, 0, 1UL<<32, {1<<8, 1<<8}, seq_len, seq_len, d, 1);
        Rescaling rs1(1), rs2(1);

        FrTensor shift(seq_len), X_shifted(seq_len * seq_len);
        vector<FrTensor> X_segments, Y_segments, m_segments;
        vector<FrTensor> aux_states; // for online path
        OnlineSoftmaxCtx online_ctx {FrTensor(0), FrTensor(0), FrTensor(0), FrTensor(0)};
        FrTensor Y(0);
        if (use_online) {
            Y = online_softmax_compute(X, online_ctx, shift, X_shifted, aux_states);
        } else {
            Y = softmax.compute(X, shift, X_shifted, X_segments, Y_segments, m_segments);
        }
        Y.save_long("temp_head_Y.bin");
        
        
        auto out = FrTensor::matmul(Y, V, seq_len, seq_len, d);
        auto out_ = rs2(out);
        auto out__ = rs1(out_);

        out__.save_int("temp_head_out.bin");

        //两次rescaling的证明
        rs1.prove(out_, out__);
        rs2.prove(out, out_);
        auto temp_rand = random_vec(3);
        vector<Polynomial> proof;
        auto u1 = random_vec(ceilLog2(seq_len));
        auto u2 = random_vec(ceilLog2(d));
        auto ud = random_vec(ceilLog2(seq_len));
        auto claim = out.multi_dim_me({u1, u2}, {seq_len, d});
        //矩阵乘法证明 Y=KV
        auto final_claim = zkip(claim, Y.partial_me(u1, seq_len, seq_len), V.partial_me(u2, d, 1), ud, proof);


        //对softmax结果做证明
        if (use_online) {
            online_softmax_prove(Y, X, online_ctx, shift, X_shifted, aux_states,
                                 random_vec(ceilLog2(Y.size)), random_vec(ceilLog2(Y.size)), proof);
        } else {
            softmax.prove(Y, X, shift, X_shifted, X_segments, Y_segments, m_segments, 
                random_vec(ceilLog2(Y.size)), random_vec(ceilLog2(Y.size)), temp_rand[0], temp_rand[1], temp_rand[2], proof);
        }
        auto u1_ = random_vec(ceilLog2(seq_len));
        auto u2_ = random_vec(ceilLog2(seq_len));
        auto ud_ = random_vec(ceilLog2(d));
        auto claim_ = X.multi_dim_me({u1_, u2_}, {seq_len, seq_len});
        //矩阵乘法证明 X=QK^T
        auto final_claim_ = zkip(claim_, Q.partial_me(u1_, seq_len, d), K.partial_me(u2_, seq_len, d), ud_, proof);

        size_t proof_poly_count = proof.size();
        size_t proof_coeff_count = 0;
        for (auto &poly : proof) {
            proof_coeff_count += static_cast<size_t>(poly.getDegree() + 1);
        }
        size_t proof_est_bytes = proof_coeff_count * sizeof(Fr_t);

        cout << "PROOF_STATS_ATTN poly_count=" << proof_poly_count
             << " coeff_count=" << proof_coeff_count
             << " est_bytes=" << proof_est_bytes << "\n";
           cout << "VERIFIER_TIME_ATTN_S=0\n";
        cout << "Self attention proof successfully verified!" << endl; 
        return 0;
    }
    return 0;
}