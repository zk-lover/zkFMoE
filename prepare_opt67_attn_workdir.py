import argparse
import os
import shutil
import subprocess
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def ensure_binaries(repo_root: Path) -> None:
    if not (repo_root / "ppgen").exists():
        run(["make", "ppgen"])
    if not (repo_root / "commit-param").exists():
        run(["make", "commit-param"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare OPT-6.7B self-attn q/k/v int+commitment files")
    parser.add_argument("--src_dir", default="../zkllm-benchmark/opt-int", help="Directory with exported int32 OPT weights")
    parser.add_argument("--out_dir", default="./zkllm-workdir/OPT-6.7b", help="Output zkllm workdir")
    parser.add_argument("--layers", type=int, default=32)
    parser.add_argument("--embed_dim", type=int, default=4096)
    parser.add_argument("--log_off_factor", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    os.chdir(repo_root)

    src_dir = (repo_root / args.src_dir).resolve()
    out_dir = (repo_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not src_dir.exists():
        raise FileNotFoundError(f"src_dir not found: {src_dir}")

    ensure_binaries(repo_root)

    pp_size = args.embed_dim << args.log_off_factor
    pp_files = {
        "q": out_dir / "self_attn.q_proj.weight-pp.bin",
        "k": out_dir / "self_attn.k_proj.weight-pp.bin",
        "v": out_dir / "self_attn.v_proj.weight-pp.bin",
    }

    for k, pp_path in pp_files.items():
        if args.overwrite or not pp_path.exists():
            run(["./ppgen", str(pp_size), str(pp_path)])

    for layer in range(args.layers):
        for proj in ("q", "k", "v"):
            src_name = f"model_decoder_layers_{layer}_self_attn_{proj}_proj_weight.bin"
            src_path = src_dir / src_name
            if not src_path.exists():
                raise FileNotFoundError(f"missing source weight: {src_path}")

            dst_int = out_dir / f"layer-{layer}-self_attn.{proj}_proj.weight-int.bin"
            dst_com = out_dir / f"layer-{layer}-self_attn.{proj}_proj.weight-commitment.bin"

            if args.overwrite or not dst_int.exists():
                shutil.copyfile(src_path, dst_int)

            if args.overwrite or not dst_com.exists():
                run([
                    "./commit-param",
                    str(pp_files[proj]),
                    str(dst_int),
                    str(dst_com),
                    str(args.embed_dim),
                    str(args.embed_dim),
                ])

    print("[OK] Prepared OPT-6.7b attention workdir:", out_dir)


if __name__ == "__main__":
    main()
