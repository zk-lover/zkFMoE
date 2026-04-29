# zkFMoE

`zkFMoE` is a cleaned open-source release of a CUDA-based zero-knowledge proving pipeline for sparse LLM inference, with a focus on the `RoPE + FlashMLA + Top-k MoE` setting. The repository keeps the original proving kernels and model-preparation scripts, while removing build products, logs, generated binaries, temporary workdirs, and local experiment artifacts.

This repository is intentionally small because it is a source-only release. It does not bundle model weights, committed parameter binaries, benchmark logs, generated figures, or prepared workdirs.

The current codebase contains:

- CUDA implementations for field arithmetic, commitments, proof generation, self-attention, FlashMLA-style attention, online softmax, and Top-k MoE lookup arguments.
- Weight preparation scripts for `OPT-6.7B`, `OPT-13B`, `LLaMA-2-13B`, `Qwen3-32B`, `DeepSeek-V2`, and `DeepSeek-V3.2`.
- Benchmark scripts for reproducing multi-model layerwise measurements and DeepSeek-V2 single-layer attention/MoE measurements.
- The earlier `zkLLM`-style demo scripts for LLaMA components, preserved for reference.

## 1. Repository Layout

```text
zkFMoE/
|-- *.cu / *.cuh / *.cpp / *.hpp   # CUDA/C++ proving kernels and helpers
|-- Makefile                       # build entry
|-- prepare_*_attn_workdir.py      # model-specific weight export + commitment preparation
|-- run_opt67_attn_benchmark.py    # generic layerwise benchmark driver
|-- plot_*.py                      # plotting templates used for figures
|-- tools/
|   |-- zkonline_softmax.py        # prototype helper for online softmax
|   `-- zk_sparse_lookup.py        # prototype helper for sparse Top-k lookup
`-- README.md
```

This release intentionally does not include:

- model weights
- generated `*.bin` files
- `zkllm-workdir/`
- benchmark logs
- figures and tables
- temporary test outputs

## 0. Model Weights Notice

Users must obtain model weights by themselves before running the preparation and benchmark scripts.

This repository does not redistribute any upstream model checkpoints for two reasons:

- the checkpoints are too large for a normal source repository
- the original model licenses usually require users to fetch the weights from the official provider

In practice, this means:

- for `OPT-13B`, `Qwen3-32B`, `DeepSeek-V2`, and `DeepSeek-V3.2`, users should download the original local model files first, then point `prepare_*_attn_workdir.py` to those local directories
- for `LLaMA-2-13B`, users should first obtain the official sharded checkpoint files and `params.json`
- for `OPT-6.7B`, the current script expects already-exported local `int32` q/k/v weight binaries, so users need to prepare those files separately before running `prepare_opt67_attn_workdir.py`

If you plan to publish this repository publicly, it is recommended to keep the release source-only and let users fetch the weights from the corresponding official model pages.

## 2. Recommended Environment

The code is CUDA-first and should be run on Linux with an NVIDIA GPU. The `Makefile` uses POSIX shell syntax and was not designed for native Windows builds.

Recommended environment:

- Ubuntu 20.04 or 22.04
- NVIDIA GPU with recent driver
- CUDA 12.1
- Python 3.10 or 3.11
- PyTorch with CUDA support

Example setup:

```bash
conda create -n zkfmoe python=3.11 -y
conda activate zkfmoe
conda install cuda -c nvidia/label/cuda-12.1.0 -y
pip install torch torchvision torchaudio
pip install numpy transformers safetensors datasets matplotlib
```

Check that the environment is healthy before running:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

## 3. Build

Before compiling, open `Makefile` and set `ARCH` to your GPU compute capability. Example:

- `sm_86` for RTX A6000 / RTX 30xx
- `sm_89` for RTX 40xx
- `sm_90` for H100

Build the main binaries:

```bash
make ppgen commit-param self-attn moe-bench
```

Or build everything listed by the current `Makefile`:

```bash
make all
```

Useful cleanup:

```bash
make clean
```

## 4. Runtime Artifacts and Metrics

Most experiments in this repository follow the same pattern:

1. Prepare a model-specific `workdir` with committed weights.
2. Run `run_opt67_attn_benchmark.py` with model-specific dimensions.
3. Capture stdout and memory usage into a log.
4. Extract the reported metrics.

The benchmark driver prints the following keys:

- `prover_pure_est_s`: pure prover time estimate
- `verifier_total_time_s`: total verifier time
- `attn_est_bytes_total`: estimated attention proof size in bytes
- `proof_est_bytes_total`: estimated total proof size in bytes when MoE is enabled
- `attn_online_total_s`: attention proving time only
- `moe_total_time_s`: MoE proving time only
- `verifier_attn_time_s`: attention verifier time only
- `verifier_moe_time_s`: MoE verifier time only

Peak memory is not emitted by the Python driver itself. To reproduce the `MAX_RSS_KB`-style number, wrap the command with `/usr/bin/time -v`:

```bash
mkdir -p results
/usr/bin/time -v python run_opt67_attn_benchmark.py ... 2>&1 | tee results/example.log
```

Then read `Maximum resident set size` from the log. If you want the exact `MAX_RSS_KB=...` suffix used in the stored experiment notes, append it manually:

```bash
rss_kb=$(grep "Maximum resident set size" results/example.log | awk '{print $6}')
echo "MAX_RSS_KB=${rss_kb}" | tee -a results/example.log
```

## 5. Preparing Model Workdirs

All `prepare_*_attn_workdir.py` scripts do the same high-level job:

- load model weights from local storage
- quantize target attention weights to `int32`
- generate public parameters with `ppgen`
- create commitments with `commit-param`
- write all outputs into a clean `workdir`

### 5.1 OPT-6.7B

`prepare_opt67_attn_workdir.py` expects already-exported `int32` q/k/v weights. By default it looks for files such as:

```text
model_decoder_layers_<layer>_self_attn_q_proj_weight.bin
model_decoder_layers_<layer>_self_attn_k_proj_weight.bin
model_decoder_layers_<layer>_self_attn_v_proj_weight.bin
```

Example:

```bash
python prepare_opt67_attn_workdir.py \
  --src_dir /path/to/opt67_qkv_int_bins \
  --out_dir ./zkllm-workdir/OPT-6.7b \
  --layers 32 \
  --embed_dim 4096 \
  --overwrite
```

### 5.2 OPT-13B

This script reads local Hugging Face shards from `pytorch_model.bin.index.json` plus `pytorch_model-*.bin`.

```bash
python prepare_opt13_attn_workdir.py \
  --model_dir /path/to/opt-13b_weights \
  --out_dir ./zkllm-workdir/OPT-13b \
  --layers 40 \
  --embed_dim 5120 \
  --overwrite
```

### 5.3 LLaMA-2-13B

This script expects the original two-shard layout with `consolidated.00.pth`, `consolidated.01.pth`, and `params.json`.

```bash
python prepare_llama13_attn_workdir.py \
  --model_dir /path/to/llama-2-13b \
  --out_dir ./zkllm-workdir/Llama-2-13b \
  --overwrite
```

### 5.4 Qwen3-32B

This script reads `model.safetensors.index.json` and the corresponding `model-*.safetensors` shards.

```bash
python prepare_qwen32_attn_workdir.py \
  --model_dir /path/to/Qwen3-32B \
  --out_dir ./zkllm-workdir/Qwen3-32b \
  --overwrite
```

At the end, the script prints the discovered benchmark arguments, including `--q_dim` and `--kv_dim`. Reuse those numbers when running the benchmark.

### 5.5 DeepSeek-V2 and DeepSeek-V3.2

Despite the filename, `prepare_deepseek_v32_attn_workdir.py` can be used for both `DeepSeek-V2` and `DeepSeek-V3.2` as long as the model follows the same MLA-style attention tensor layout.

DeepSeek-V2:

```bash
python prepare_deepseek_v32_attn_workdir.py \
  --model_dir /path/to/DeepSeek-V2 \
  --out_dir ./zkllm-workdir/DeepSeek-V2 \
  --overwrite
```

DeepSeek-V3.2:

```bash
python prepare_deepseek_v32_attn_workdir.py \
  --model_dir /path/to/DeepSeek-V3.2 \
  --out_dir ./zkllm-workdir/DeepSeek-V3.2 \
  --overwrite
```

The script prints the recommended benchmark arguments, including `--layers`, `--embed_dim`, and `--linear_mode linear_deepseek_mla`.

## 6. Reproducing the Six-Model Four-Metric Evaluation

The six-model evaluation mentioned in this project covers:

- `OPT-6.7B`
- `OPT-13B`
- `LLaMA-2-13B`
- `Qwen3-32B`
- `DeepSeek-V2`
- `DeepSeek-V3.2`

The four metrics to collect are:

1. prover time: `prover_pure_est_s`
2. proof size: `proof_est_bytes_total` or `attn_est_bytes_total`
3. verifier time: `verifier_total_time_s`
4. peak memory: `MAX_RSS_KB` from `/usr/bin/time -v`

Create a result directory first:

```bash
mkdir -p results
```

### 6.1 OPT-6.7B

```bash
/usr/bin/time -v python run_opt67_attn_benchmark.py \
  --layers 32 \
  --seq_len 2048 \
  --embed_dim 4096 \
  --workdir ./zkllm-workdir/OPT-6.7b \
  --input_file ./results/opt67-input-seq2048-int.bin \
  --with_moe \
  --moe_experts 16 \
  --moe_topk 2 \
  2>&1 | tee results/opt67_seq2048.log
```

### 6.2 OPT-13B

```bash
/usr/bin/time -v python run_opt67_attn_benchmark.py \
  --layers 40 \
  --seq_len 2048 \
  --embed_dim 5120 \
  --workdir ./zkllm-workdir/OPT-13b \
  --input_file ./results/opt13-input-seq2048-int.bin \
  --with_moe \
  --moe_experts 16 \
  --moe_topk 2 \
  2>&1 | tee results/opt13_seq2048.log
```

### 6.3 LLaMA-2-13B

```bash
/usr/bin/time -v python run_opt67_attn_benchmark.py \
  --layers 40 \
  --seq_len 2048 \
  --embed_dim 5120 \
  --workdir ./zkllm-workdir/Llama-2-13b \
  --input_file ./results/llama13-input-seq2048-int.bin \
  2>&1 | tee results/llama13_seq2048.log
```

### 6.4 Qwen3-32B

Replace `<layers>`, `<embed_dim>`, `<q_dim>`, and `<kv_dim>` with the values printed by `prepare_qwen32_attn_workdir.py`.

```bash
/usr/bin/time -v python run_opt67_attn_benchmark.py \
  --layers <layers> \
  --seq_len 2048 \
  --embed_dim <embed_dim> \
  --q_dim <q_dim> \
  --kv_dim <kv_dim> \
  --workdir ./zkllm-workdir/Qwen3-32b \
  --input_file ./results/qwen32-input-seq2048-int.bin \
  2>&1 | tee results/qwen32_seq2048.log
```

### 6.5 DeepSeek-V2

Replace `<layers>` and `<embed_dim>` with the values printed by `prepare_deepseek_v32_attn_workdir.py`.

```bash
/usr/bin/time -v python run_opt67_attn_benchmark.py \
  --layers <layers> \
  --seq_len 2048 \
  --embed_dim <embed_dim> \
  --linear_mode linear_deepseek_mla \
  --workdir ./zkllm-workdir/DeepSeek-V2 \
  --input_file ./results/deepseekv2-input-seq2048-int.bin \
  2>&1 | tee results/deepseek_v2_seq2048.log
```

### 6.6 DeepSeek-V3.2

```bash
/usr/bin/time -v python run_opt67_attn_benchmark.py \
  --layers <layers> \
  --seq_len 2048 \
  --embed_dim <embed_dim> \
  --linear_mode linear_deepseek_mla \
  --workdir ./zkllm-workdir/DeepSeek-V3.2 \
  --input_file ./results/deepseekv32-input-seq2048-int.bin \
  2>&1 | tee results/deepseek_v32_seq2048.log
```

### 6.7 How to Read the Logs

For the six-model table, read the following keys from each log:

- prover time: `prover_pure_est_s`
- verifier time: `verifier_total_time_s`
- proof size:
  - use `proof_est_bytes_total` when `--with_moe` is enabled
  - otherwise use `attn_est_bytes_total`
- peak memory:
  - read `Maximum resident set size`
  - or append `MAX_RSS_KB=<value>` yourself as shown above

## 7. Reproducing DeepSeek-V2 Single-Layer Attention and Top-k MoE Proof-Time Curves

This experiment measures `seq_len in {256, 512, 1024, 2048, 4096}` for a single layer of `DeepSeek-V2`.

### 7.1 Prepare the DeepSeek-V2 workdir

```bash
python prepare_deepseek_v32_attn_workdir.py \
  --model_dir /path/to/DeepSeek-V2 \
  --out_dir ./zkllm-workdir/DeepSeek-V2 \
  --overwrite
```

Assume the script reports the correct `<embed_dim>` for the model.

### 7.2 Run the five sequence lengths

Run the following loop:

```bash
mkdir -p results
for SEQ in 256 512 1024 2048 4096; do
  /usr/bin/time -v python run_opt67_attn_benchmark.py \
    --layers 1 \
    --seq_len ${SEQ} \
    --embed_dim <embed_dim> \
    --linear_mode linear_deepseek_mla \
    --workdir ./zkllm-workdir/DeepSeek-V2 \
    --input_file ./results/deepseekv2-input-seq${SEQ}-int.bin \
    --with_moe \
    --moe_experts 16 \
    --moe_topk 2 \
    2>&1 | tee results/deepseek_v2_singlelayer_seq${SEQ}.log
done
```

### 7.3 Extract the single-layer metrics

For each `results/deepseek_v2_singlelayer_seq${SEQ}.log`, collect:

- attention proving time: `attn_online_total_s`
- attention proof size: `attn_est_bytes_total / 1024`
- attention verifying time: `verifier_attn_time_s`
- MoE proving time: `moe_total_time_s`
- MoE proof size: `moe_est_bytes_total / 1024`
- MoE verifying time: `verifier_moe_time_s`

These six numbers are enough to reconstruct a table like:

```text
seq_len  attn_proving_time_s  attn_proof_size_kb  attn_verifying_time_s  moe_proving_time_s  moe_proof_size_kb  moe_verifying_time_s
```

### 7.4 Plotting

`plot_deepseek_v2_singlelayer_proof_time.py` is a plotting template used for the final figures. It currently contains hardcoded arrays rather than automatically parsing logs. After collecting your five rows of results, update the arrays in that script and run:

```bash
python plot_deepseek_v2_singlelayer_proof_time.py
```

This produces:

- `deepseek_v2_singlelayer_attn_proof_time_bar.png`
- `deepseek_v2_singlelayer_moe_proof_time_bar.png`

## 8. Reproducing the Prover-Time Comparison Figure

After running the six-model evaluation, fill the `this_work` array in `plot_prover_time_comparison.py` with the `prover_pure_est_s` values extracted from your logs, then run:

```bash
python plot_prover_time_comparison.py
```

Note:

- the current plotting file is a template and uses hardcoded numbers
- the font path in the script assumes a Linux environment with Noto CJK installed
- adjust the output path in the script if needed

## 9. Additional Notes

- `run_opt67_attn_benchmark.py` is now the generic benchmark driver for multiple models, despite its historical filename.
- If the `--input_file` does not exist, the benchmark driver auto-generates a random `int32` input tensor with a fixed RNG seed.
- `prepare_qwen32_attn_workdir.py` and `prepare_deepseek_v32_attn_workdir.py` infer critical dimensions from the local model files. Prefer using the printed benchmark arguments rather than manually guessing them.
- The current release focuses on reproducibility of the benchmark pipeline, not on packaging a polished training/inference framework.
- Because this is a CUDA-heavy proving prototype, exact timings depend on GPU model, driver version, and whether other GPU workloads are running.

## 10. License

This repository keeps the original `LICENSE` file from the source project. Please check it before redistributing derived releases.
