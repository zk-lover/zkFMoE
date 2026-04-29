import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch


def run(cmd: List[str]) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def ensure_binaries(repo_root: Path) -> None:
    if not (repo_root / "ppgen").exists():
        run(["make", "ppgen"])
    if not (repo_root / "commit-param").exists():
        run(["make", "commit-param"])


def save_int_bin_from_weight(weight: torch.Tensor, path: Path, log_scaling_factor: int) -> Tuple[int, int]:
    # Expect weight in (out_dim, in_dim), convert to (in_dim, out_dim) layout used by current pipeline.
    w = weight.float().t().contiguous()
    in_dim, out_dim = int(w.shape[0]), int(w.shape[1])
    scale = float(1 << log_scaling_factor)
    arr = torch.round(w * scale).to(torch.int32).cpu().numpy().astype(np.int32, copy=False)
    arr.tofile(str(path))
    return in_dim, out_dim


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare LLaMA-2-13B attention q/k/v int+commitment files from consolidated shards")
    parser.add_argument("--model_dir", default="/disk1/daqi/models/llama-2-13b")
    parser.add_argument("--out_dir", default="/disk1/daqi/zkllm-workdir/Llama-2-13b")
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--embed_dim", type=int, default=None)
    parser.add_argument("--log_off_factor", type=int, default=5)
    parser.add_argument("--log_scaling_factor", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    os.chdir(repo_root)

    model_dir = Path(args.model_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    params = json.loads((model_dir / "params.json").read_text())
    n_layers = int(params["n_layers"])
    embed_dim = int(params["dim"])
    if args.layers is not None:
        n_layers = min(n_layers, args.layers)
    if args.embed_dim is not None:
        embed_dim = args.embed_dim

    ensure_binaries(repo_root)

    ckpt0_path = model_dir / "consolidated.00.pth"
    ckpt1_path = model_dir / "consolidated.01.pth"
    print("[LOAD]", ckpt0_path)
    ckpt0 = torch.load(ckpt0_path, map_location="cpu")
    print("[LOAD]", ckpt1_path)
    ckpt1 = torch.load(ckpt1_path, map_location="cpu")

    pp_size = embed_dim << args.log_off_factor
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

    print(f"[INFO] preparing layers={n_layers}, embed_dim={embed_dim}")
    for layer in range(n_layers):
        for proj in ("q", "k", "v"):
            key = f"layers.{layer}.attention.w{proj}.weight"
            if key not in ckpt0 or key not in ckpt1:
                raise KeyError(f"Missing key in shard(s): {key}")

            # LLaMA-2-13B is tensor-parallel sharded along output dimension for wq/wk/wv.
            weight = torch.cat([ckpt0[key], ckpt1[key]], dim=0)

            dst_int = out_dir / f"layer-{layer}-self_attn.{proj}_proj.weight-int.bin"
            dst_com = out_dir / f"layer-{layer}-self_attn.{proj}_proj.weight-commitment.bin"

            if args.overwrite or not dst_int.exists():
                in_dim, out_dim = save_int_bin_from_weight(weight, dst_int, args.log_scaling_factor)
            else:
                in_dim, out_dim = embed_dim, embed_dim

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

    print(f"[OK] prepared LLaMA-2-13b attention workdir: {out_dir}")
    print(f"[OK] benchmark args: --layers {n_layers} --embed_dim {embed_dim} --workdir {out_dir}")


if __name__ == "__main__":
    main()
