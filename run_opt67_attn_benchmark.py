import argparse
import os
import subprocess
import time
from pathlib import Path
import re
import numpy as np
from typing import List, Tuple


def run_cmd(cmd: List[str]) -> float:
    t0 = time.perf_counter()
    proc = subprocess.run(cmd)
    t1 = time.perf_counter()
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")
    return t1 - t0


def run_cmd_capture(cmd: List[str]) -> Tuple[float, str, str]:
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    t1 = time.perf_counter()
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr}\n{proc.stdout}")
    return t1 - t0, proc.stdout, proc.stderr


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OPT-6.7B self-attention proof benchmark layer-by-layer")
    parser.add_argument("--layers", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--embed_dim", type=int, default=4096)
    parser.add_argument("--q_dim", type=int, default=None)
    parser.add_argument("--kv_dim", type=int, default=None)
    parser.add_argument("--workdir", default="./zkllm-workdir/OPT-6.7b")
    parser.add_argument("--input_file", default="./opt67-input-seq16-int.bin")
    parser.add_argument("--output_file", default="./opt67-attn-output.bin")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--attn_mode", choices=["online", "flash_online"], default="online")
    parser.add_argument("--linear_mode", choices=["linear", "linear_deepseek_mla", "linear_deepseek_v32"], default="linear")
    parser.add_argument("--with_moe", action="store_true", help="Run MoE benchmark after each layer")
    parser.add_argument("--moe_experts", type=int, default=16)
    parser.add_argument("--moe_topk", type=int, default=2)
    parser.add_argument("--moe_logits_file", default="./temp_moe_logits.bin")
    parser.add_argument("--moe_expert_out_file", default="./temp_moe_expert_out.bin")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    os.chdir(repo_root)

    if args.rebuild or not (repo_root / "self-attn").exists():
        subprocess.run(["make", "self-attn"], check=True)
    if args.with_moe and (args.rebuild or not (repo_root / "moe-bench").exists()):
        subprocess.run(["make", "moe-bench"], check=True)

    input_path = (repo_root / args.input_file).resolve()
    if not input_path.exists():
        rng = np.random.default_rng(20260302)
        arr = rng.integers(low=-16, high=16, size=(args.seq_len * args.embed_dim,), dtype=np.int32)
        arr.tofile(str(input_path))
        print("[OK] generated input:", input_path)

    linear_total = 0.0
    attn_total = 0.0
    moe_total = 0.0
    attn_poly_total = 0
    attn_coeff_total = 0
    attn_bytes_total = 0
    moe_poly_total = 0
    moe_coeff_total = 0
    moe_bytes_total = 0
    moe_pairwise_total = 0
    verifier_total = 0.0
    verifier_linear_total = 0.0
    verifier_attn_total = 0.0
    verifier_moe_total = 0.0

    t_all0 = time.perf_counter()
    for layer in range(args.layers):
        layer_prefix = f"layer-{layer}"
        print(f"\\n===== layer {layer} / {args.layers - 1} =====")

        cmd_linear = [
            "./self-attn",
            args.linear_mode,
            str(input_path),
            str(args.seq_len),
            str(args.embed_dim),
            args.workdir,
            layer_prefix,
            args.output_file,
        ]
        if args.q_dim is not None:
            cmd_linear.append(str(args.q_dim))
        if args.kv_dim is not None:
            if args.q_dim is None:
                cmd_linear.append(str(args.embed_dim))
            cmd_linear.append(str(args.kv_dim))
        dt_linear, linear_stdout, linear_stderr = run_cmd_capture(cmd_linear)
        linear_total += dt_linear
        linear_text = f"{linear_stdout}\n{linear_stderr}"
        m_v_linear_all = re.findall(r"VERIFIER_TIME_LINEAR_S=([0-9eE+\-.]+)", linear_text)
        if m_v_linear_all:
            v_linear = sum(float(x) for x in m_v_linear_all)
            verifier_linear_total += v_linear
            verifier_total += v_linear
        print(f"[TIME] linear layer {layer}: {dt_linear:.3f}s")

        attn_binary_mode = "attn_online" if args.attn_mode == "online" else "attn_flash_online"
        cmd_attn = [
            "./self-attn",
            attn_binary_mode,
            "unused",
            str(args.seq_len),
            str(args.embed_dim),
            args.workdir,
            layer_prefix,
            args.output_file,
        ]
        dt_attn, attn_stdout, attn_stderr = run_cmd_capture(cmd_attn)
        attn_total += dt_attn
        attn_text = f"{attn_stdout}\n{attn_stderr}"
        m_attn = re.search(r"PROOF_STATS_ATTN\s+poly_count=(\d+)\s+coeff_count=(\d+)\s+est_bytes=(\d+)", attn_text)
        if m_attn is not None:
            attn_poly_total += int(m_attn.group(1))
            attn_coeff_total += int(m_attn.group(2))
            attn_bytes_total += int(m_attn.group(3))
        m_v_attn = re.search(r"VERIFIER_TIME_ATTN_S=([0-9eE+\-.]+)", attn_text)
        if m_v_attn is not None:
            v_attn = float(m_v_attn.group(1))
            verifier_attn_total += v_attn
            verifier_total += v_attn
        print(f"[TIME] attn_online layer {layer}: {dt_attn:.3f}s")

        if args.with_moe:
            head_out_path = repo_root / "temp_head_out.bin"
            if not head_out_path.exists():
                raise FileNotFoundError("temp_head_out.bin not found after self-attn run")
            head_out = np.fromfile(str(head_out_path), dtype=np.int32)
            if args.seq_len <= 0 or head_out.size % args.seq_len != 0:
                raise RuntimeError(
                    f"temp_head_out size mismatch: got {head_out.size}, not divisible by seq_len={args.seq_len}"
                )
            head_dim = head_out.size // args.seq_len
            need_dim = 2 * args.moe_experts
            if head_dim < need_dim:
                raise RuntimeError(
                    f"temp_head_out has insufficient dim for MoE slicing: head_dim={head_dim}, need_at_least={need_dim}"
                )
            head_out = head_out.reshape(args.seq_len, head_dim)
            moe_logits = head_out[:, :args.moe_experts].reshape(-1)
            moe_expert_out = head_out[:, args.moe_experts:2 * args.moe_experts].reshape(-1)
            moe_logits.tofile(str((repo_root / args.moe_logits_file).resolve()))
            moe_expert_out.tofile(str((repo_root / args.moe_expert_out_file).resolve()))

            cmd_moe = [
                "./moe-bench",
                args.moe_logits_file,
                args.moe_expert_out_file,
                str(args.seq_len),
                str(args.moe_experts),
                str(args.moe_topk),
            ]
            t0 = time.perf_counter()
            proc = subprocess.run(cmd_moe, capture_output=True, text=True)
            t1 = time.perf_counter()
            if proc.returncode != 0:
                raise RuntimeError(f"moe-bench failed: {proc.stderr}\n{proc.stdout}")
            dt_moe = t1 - t0
            m = re.search(r"moe_total_time_s\s*:\s*([0-9eE+\-.]+)", proc.stdout)
            if m is not None:
                dt_moe = float(m.group(1))
            m_v_moe = re.search(r"verifier_time_s\s*:\s*([0-9eE+\-.]+)", proc.stdout)
            if m_v_moe is not None:
                v_moe = float(m_v_moe.group(1))
                verifier_moe_total += v_moe
                verifier_total += v_moe
            m_moe = re.search(r"PROOF_STATS_MOE\s+poly_count=(\d+)\s+coeff_count=(\d+)\s+est_bytes=(\d+)", proc.stdout)
            if m_moe is not None:
                moe_poly_total += int(m_moe.group(1))
                moe_coeff_total += int(m_moe.group(2))
                moe_bytes_total += int(m_moe.group(3))
            m_pair = re.search(r"pairwise_checks\s*:\s*(\d+)", proc.stdout)
            if m_pair is not None:
                moe_pairwise_total += int(m_pair.group(1))
            moe_total += dt_moe
            print(f"[TIME] moe layer {layer}: {dt_moe:.3f}s")

    t_all1 = time.perf_counter()
    total = t_all1 - t_all0

    print("\\n========== OPT-6.7B ATTN BENCH SUMMARY ==========")
    print(f"layers                 : {args.layers}")
    print(f"seq_len                : {args.seq_len}")
    print(f"embed_dim              : {args.embed_dim}")
    print(f"linear_total_time_s    : {linear_total:.6f}")
    print(f"attn_online_total_s    : {attn_total:.6f}")
    if args.with_moe:
        print(f"moe_total_time_s       : {moe_total:.6f}")
        print(f"block_e2e_total_s      : {linear_total + attn_total + moe_total:.6f}")
    print(f"verifier_total_time_s  : {verifier_total:.6f}")
    print(f"verifier_linear_time_s : {verifier_linear_total:.6f}")
    print(f"verifier_attn_time_s   : {verifier_attn_total:.6f}")
    if args.with_moe:
        print(f"verifier_moe_time_s    : {verifier_moe_total:.6f}")
    prover_pure_est = (linear_total + attn_total + moe_total) - verifier_total if args.with_moe else (linear_total + attn_total) - verifier_total
    if prover_pure_est < 0:
        prover_pure_est = 0.0
    print(f"prover_pure_est_s      : {prover_pure_est:.6f}")
    if args.layers > 0:
        print(f"verifier_agg_est_s     : {verifier_total / args.layers:.6f}")
    print(f"end_to_end_total_s     : {total:.6f}")

    print("\n========== PROOF ARTIFACT SUMMARY ==========")
    print(f"attn_poly_total        : {attn_poly_total}")
    print(f"attn_coeff_total       : {attn_coeff_total}")
    print(f"attn_est_bytes_total   : {attn_bytes_total}")
    if args.with_moe:
        print(f"moe_poly_total         : {moe_poly_total}")
        print(f"moe_coeff_total        : {moe_coeff_total}")
        print(f"moe_est_bytes_total    : {moe_bytes_total}")
        print(f"moe_pairwise_checks    : {moe_pairwise_total}")
        print(f"proof_est_bytes_total  : {attn_bytes_total + moe_bytes_total}")


if __name__ == "__main__":
    main()
