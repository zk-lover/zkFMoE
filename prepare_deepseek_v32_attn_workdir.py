import argparse
import gc
import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from safetensors.torch import load_file
from safetensors import safe_open


BLOCK = 128


def run(cmd: List[str]) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def ensure_binaries(repo_root: Path) -> None:
    if not (repo_root / "ppgen").exists():
        run(["make", "ppgen"])
    if not (repo_root / "commit-param").exists():
        run(["make", "commit-param"])


def expand_scale(scale: torch.Tensor, out_dim: int, in_dim: int) -> torch.Tensor:
    expanded = scale.repeat_interleave(BLOCK, dim=0).repeat_interleave(BLOCK, dim=1)
    return expanded[:out_dim, :in_dim]


def dequant_fp8(weight: torch.Tensor, scale_inv: torch.Tensor) -> torch.Tensor:
    # Model stores block-wise inverse scales for FP8 weights.
    s = expand_scale(scale_inv.float(), weight.shape[0], weight.shape[1])
    return weight.float() * s


def materialize_weight(weight: torch.Tensor, scale_inv: Optional[torch.Tensor]) -> torch.Tensor:
    if scale_inv is None:
        return weight.float()
    return dequant_fp8(weight, scale_inv)


def save_int_bin_from_weight(weight: torch.Tensor, path: Path, log_scaling_factor: int) -> Tuple[int, int]:
    w = weight.float().t().contiguous()
    in_dim, out_dim = int(w.shape[0]), int(w.shape[1])
    scale = float(1 << log_scaling_factor)
    arr = torch.round(w * scale).to(torch.int32).cpu().numpy().astype(np.int32, copy=False)
    arr.tofile(str(path))
    return in_dim, out_dim


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare DeepSeek MLA weights for self-attn benchmark")
    parser.add_argument("--model_dir", default="/disk1/daqi/models/deepseek-ai/DeepSeek-V3.2")
    parser.add_argument("--out_dir", default="./zkllm-workdir/DeepSeek-V3.2")
    parser.add_argument("--layers", type=int, default=None, help="Limit prepared layers; default uses config")
    parser.add_argument("--log_off_factor", type=int, default=5)
    parser.add_argument("--log_scaling_factor", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    os.chdir(repo_root)

    model_dir = Path(args.model_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not model_dir.exists():
        raise FileNotFoundError(f"model_dir not found: {model_dir}")

    cfg = json.loads((model_dir / "config.json").read_text())
    hidden_size = int(cfg["hidden_size"])
    n_layers = int(cfg["num_hidden_layers"])
    if args.layers is not None:
        n_layers = min(n_layers, args.layers)

    ensure_binaries(repo_root)

    idx = json.loads((model_dir / "model.safetensors.index.json").read_text())
    weight_map: Dict[str, str] = idx["weight_map"]

    def keys_for_layer(layer: int) -> List[Tuple[str, Optional[str]]]:
        base = f"model.layers.{layer}.self_attn"
        has_scale = f"{base}.q_a_proj.weight_scale_inv" in weight_map
        return [
            (f"{base}.q_a_proj.weight", f"{base}.q_a_proj.weight_scale_inv" if has_scale else None),
            (f"{base}.q_b_proj.weight", f"{base}.q_b_proj.weight_scale_inv" if has_scale else None),
            (f"{base}.kv_a_proj_with_mqa.weight", f"{base}.kv_a_proj_with_mqa.weight_scale_inv" if has_scale else None),
            (f"{base}.kv_b_proj.weight", f"{base}.kv_b_proj.weight_scale_inv" if has_scale else None),
        ]

    for layer in range(n_layers):
        for wk, sk in keys_for_layer(layer):
            if wk not in weight_map or (sk is not None and sk not in weight_map):
                raise KeyError(f"Missing DeepSeek-V3.2 key: {wk} / {sk}")

    # Build pp once based on real dimensions from layer 0.
    sample_shard = model_dir / weight_map[keys_for_layer(0)[0][0]]
    sample_state = load_file(str(sample_shard), device="cpu")
    dims = {
        "q_a": tuple(sample_state[keys_for_layer(0)[0][0]].shape),
        "q_b": tuple(sample_state[keys_for_layer(0)[1][0]].shape),
        "kv_a": tuple(sample_state[keys_for_layer(0)[2][0]].shape),
        "kv_b": tuple(sample_state[keys_for_layer(0)[3][0]].shape),
    }

    pp_specs = {
        "q_a_proj": dims["q_a"],
        "q_b_proj": dims["q_b"],
        "kv_a_proj_with_mqa": dims["kv_a"],
        "kv_b_proj": dims["kv_b"],
    }
    for name, (out_dim, in_dim) in pp_specs.items():
        pp_size = max(out_dim, in_dim) << args.log_off_factor
        pp_path = out_dir / f"self_attn.{name}.weight-pp.bin"
        if args.overwrite or not pp_path.exists():
            run(["./ppgen", str(pp_size), str(pp_path)])

    print(f"[INFO] preparing DeepSeek-V3.2 layers={n_layers}, hidden_size={hidden_size}")
    map_name = {
        "q_a_proj.weight": "q_a_proj",
        "q_b_proj.weight": "q_b_proj",
        "kv_a_proj_with_mqa.weight": "kv_a_proj_with_mqa",
        "kv_b_proj.weight": "kv_b_proj",
    }

    for layer in range(n_layers):
        print(f"[LAYER] {layer}/{n_layers - 1}")
        for wkey, skey in keys_for_layer(layer):
            suffix = wkey.split("self_attn.", 1)[1]
            short = map_name[suffix]
            w_shard = model_dir / weight_map[wkey]
            with safe_open(str(w_shard), framework="pt", device="cpu") as f:
                w = f.get_tensor(wkey)
            s = None
            if skey is not None:
                s_shard = model_dir / weight_map[skey]
                with safe_open(str(s_shard), framework="pt", device="cpu") as f:
                    s = f.get_tensor(skey)
            w_deq = materialize_weight(w, s)

            dst_int = out_dir / f"layer-{layer}-self_attn.{short}.weight-int.bin"
            dst_com = out_dir / f"layer-{layer}-self_attn.{short}.weight-commitment.bin"
            pp_path = out_dir / f"self_attn.{short}.weight-pp.bin"

            if args.overwrite or not dst_int.exists():
                in_dim, out_dim = save_int_bin_from_weight(w_deq, dst_int, args.log_scaling_factor)
            else:
                in_dim = int(w.shape[1])
                out_dim = int(w.shape[0])

            if args.overwrite or not dst_com.exists():
                run([
                    "./commit-param",
                    str(pp_path),
                    str(dst_int),
                    str(dst_com),
                    str(in_dim),
                    str(out_dim),
                ])

                gc.collect()

    print(f"[OK] prepared DeepSeek MLA attn workdir: {out_dir}")
    print(f"[OK] benchmark args: --linear_mode linear_deepseek_mla --layers {n_layers} --embed_dim {hidden_size}")


if __name__ == "__main__":
    main()
