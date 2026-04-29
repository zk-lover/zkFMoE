import argparse
import gc
import json
import os
import subprocess
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from safetensors.torch import load_file


def run(cmd: List[str]) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def ensure_binaries(repo_root: Path) -> None:
    if not (repo_root / "ppgen").exists():
        run(["make", "ppgen"])
    if not (repo_root / "commit-param").exists():
        run(["make", "commit-param"])


def save_int_bin_from_weight(weight: torch.Tensor, path: Path, log_scaling_factor: int) -> Tuple[int, int]:
    if weight.ndim != 2:
        raise ValueError(f"Expected 2D weight, got shape={tuple(weight.shape)}")
    w = weight.float().t().contiguous()
    in_dim, out_dim = int(w.shape[0]), int(w.shape[1])
    scale = float(1 << log_scaling_factor)
    arr = torch.round(w * scale).to(torch.int32).cpu().numpy().astype(np.int32, copy=False)
    arr.tofile(str(path))
    return in_dim, out_dim


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Qwen3-32B self-attn q/k/v int+commitment files from local HF safetensors weights")
    parser.add_argument("--model_dir", default="../zkllm-benchmark/Qwen3-32B/Qwen3-32B", help="Qwen3-32B directory containing config.json and model-*.safetensors")
    parser.add_argument("--out_dir", default="./zkllm-workdir/Qwen3-32b", help="Output zkllm workdir")
    parser.add_argument("--layers", type=int, default=64)
    parser.add_argument("--embed_dim", type=int, default=5120)
    parser.add_argument("--log_off_factor", type=int, default=5)
    parser.add_argument("--log_scaling_factor", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    os.chdir(repo_root)

    model_dir = (repo_root / args.model_dir).resolve()
    out_dir = (repo_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not model_dir.exists():
        raise FileNotFoundError(f"model_dir not found: {model_dir}")

    config = json.loads((model_dir / "config.json").read_text())
    layers = int(config.get("num_hidden_layers", args.layers))
    embed_dim = int(config.get("hidden_size", args.embed_dim))

    if layers != args.layers:
        print(f"[WARN] args.layers={args.layers} differs from config={layers}, using config value")
    if embed_dim != args.embed_dim:
        print(f"[WARN] args.embed_dim={args.embed_dim} differs from config={embed_dim}, using config value")

    ensure_binaries(repo_root)

    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing index file: {index_path}")
    weight_map: dict[str, str] = json.loads(index_path.read_text())["weight_map"]

    needed: dict[str, tuple[int, str]] = {}
    for layer in range(layers):
        for proj in ("q", "k", "v"):
            key = f"model.layers.{layer}.self_attn.{proj}_proj.weight"
            needed[key] = (layer, proj)
            if key not in weight_map:
                raise KeyError(f"Missing key in index: {key}")

    first_q = load_file(str(model_dir / weight_map["model.layers.0.self_attn.q_proj.weight"]), device="cpu")["model.layers.0.self_attn.q_proj.weight"]
    first_k = load_file(str(model_dir / weight_map["model.layers.0.self_attn.k_proj.weight"]), device="cpu")["model.layers.0.self_attn.k_proj.weight"]
    q_dim = int(first_q.shape[0])
    kv_dim = int(first_k.shape[0])

    shard_to_keys: dict[str, list[str]] = {}
    for key in needed.keys():
        shard = weight_map[key]
        shard_to_keys.setdefault(shard, []).append(key)

    pp_size = max(q_dim, kv_dim) << args.log_off_factor
    pp_files = {
        "q": out_dir / "self_attn.q_proj.weight-pp.bin",
        "k": out_dir / "self_attn.k_proj.weight-pp.bin",
        "v": out_dir / "self_attn.v_proj.weight-pp.bin",
    }
    for proj, pp_path in pp_files.items():
        if args.overwrite or not pp_path.exists():
            run(["./ppgen", str(pp_size), str(pp_path)])
        else:
            print(f"[SKIP] existing pp: {pp_path.name}")

    print(f"[INFO] exporting q/k/v for layers={layers}, embed_dim={embed_dim}, q_dim={q_dim}, kv_dim={kv_dim}")
    for shard_name, keys in sorted(shard_to_keys.items()):
        shard_path = model_dir / shard_name
        if not shard_path.exists():
            raise FileNotFoundError(f"Missing shard: {shard_path}")

        print(f"[LOAD] {shard_name} ({len(keys)} needed tensors)")
        state = load_file(str(shard_path), device="cpu")
        for key in keys:
            layer, proj = needed[key]
            weight = state[key]
            dst_int = out_dir / f"layer-{layer}-self_attn.{proj}_proj.weight-int.bin"
            dst_com = out_dir / f"layer-{layer}-self_attn.{proj}_proj.weight-commitment.bin"

            if args.overwrite or not dst_int.exists():
                in_dim, out_dim = save_int_bin_from_weight(weight, dst_int, args.log_scaling_factor)
            else:
                in_dim = embed_dim
                out_dim = q_dim if proj == "q" else kv_dim

            if args.overwrite or not dst_com.exists():
                run([
                    "./commit-param",
                    str(pp_files[proj]),
                    str(dst_int),
                    str(dst_com),
                    str(in_dim),
                    str(out_dim),
                ])
            else:
                print(f"[SKIP] existing commitment: {dst_com.name}")

        del state
        gc.collect()

    print(f"[OK] Prepared Qwen3-32b attention workdir: {out_dir}")
    print(f"[OK] Use benchmark args: --layers {layers} --embed_dim {embed_dim} --q_dim {q_dim} --kv_dim {kv_dim} --workdir {out_dir}")


if __name__ == "__main__":
    main()
