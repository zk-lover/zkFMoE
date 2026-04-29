import os, sys
import argparse
import torch
import numpy as np

parser = argparse.ArgumentParser(description='OPT PPGen and Commit')
parser.add_argument('--model_path', type=str, default='/home/daqi/opt', help='Local path to OPT model')
parser.add_argument('--log_off_factor', type=int, default=5, help='log offset factor for ppgen')
parser.add_argument('--log_scaling_factor', type=int, default=16, help='log scaling factor for fixed-point conversion')
parser.add_argument('--pad_to', type=int, default=0, help='If >0, pad embedding-related dims to this value')

from transformers import AutoTokenizer, AutoModelForCausalLM

def save_weight_int(int_weight: torch.Tensor, path):
    if path[-4:] != '.bin':
        raise ValueError('Path must end with .bin')
    int_weight.cpu().detach().numpy().astype(np.int32).tofile(path)


if __name__ == '__main__':
    args = parser.parse_args()
    # ensure ppgen and commit-param exist; build only if missing
    repo_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(repo_root)
    if not os.path.exists(os.path.join(repo_root, 'ppgen')):
        compilation_error = os.system('make ppgen')
        if compilation_error:
            print("Error compiling ppgen")
            exit(1)
    if not os.path.exists(os.path.join(repo_root, 'commit-param')):
        compilation_error = os.system('make commit-param')
        if compilation_error:
            print("Error compiling commit-param")
            exit(1)

    model = AutoModelForCausalLM.from_pretrained(args.model_path, local_files_only=True)
    scaling_factor = 1 << args.log_scaling_factor

    outdir = f"./zkllm-workdir/OPT-125m"
    os.makedirs(outdir, exist_ok=True)

    # iterate decoder layers
    try:
        layers = model.model.decoder.layers
    except Exception:
        print('Unexpected model layout; cannot find decoder.layers')
        sys.exit(1)

    # global embed dim
    embed_dim = model.model.decoder.embed_tokens.weight.shape[1]
    for i, layer in enumerate(layers):
        for j, w in layer.named_parameters():
            orig_shape = tuple(w.shape)
            if len(w.shape) == 2:
                # PyTorch weight is [out, in]; we transpose to [in, out]
                w_orig = w.float().T
                pp_size = w.shape[0]
                pp_size <<= args.log_off_factor
            else:
                w_orig = w.float()
                (pp_size,) = w.shape

            # padding if requested: pad rows/cols of w_orig when they correspond to embed_dim
            if args.pad_to and args.pad_to > 0:
                # w_orig has shape (in, out) for 2D, or (dim,) for 1D
                if len(w_orig.shape) == 2 and args.pad_to > embed_dim:
                    in_dim, out_dim = w_orig.shape
                    need_pad_in = (in_dim == embed_dim)
                    need_pad_out = (out_dim == embed_dim)
                    # first pad rows if needed
                    if need_pad_in:
                        pad_rows = args.pad_to - in_dim
                        pad_block = torch.zeros((pad_rows, out_dim), dtype=w_orig.dtype)
                        w_orig = torch.cat([w_orig, pad_block], dim=0)
                        in_dim = args.pad_to
                    # then pad cols if needed
                    if need_pad_out:
                        pad_cols = args.pad_to - out_dim
                        pad_block2 = torch.zeros((in_dim, pad_cols), dtype=w_orig.dtype)
                        w_orig = torch.cat([w_orig, pad_block2], dim=1)
                elif len(w_orig.shape) == 1 and w_orig.shape[0] == embed_dim and args.pad_to > embed_dim:
                    pad_block = torch.zeros((args.pad_to - embed_dim,), dtype=w_orig.dtype)
                    w_orig = torch.cat([w_orig, pad_block], dim=0)

            w_out = torch.round(w_orig * scaling_factor).to(torch.int32)
            print(f'Layer {i} param {j} shape {w.shape} -> pp_size {pp_size}')

            pp_path = os.path.join(outdir, f"{j}-pp.bin")
            int_bin_path = os.path.join(outdir, f"layer-{i}-{j}-int.bin")
            commitment_path = os.path.join(outdir, f"layer-{i}-{j}-commitment.bin")

            # generate pp only if not exists
            if not os.path.exists(pp_path):
                os.system(f'./ppgen {pp_size} {pp_path}')

            save_weight_int(w_out, int_bin_path)

            if len(w_out.shape) == 2:
                os.system(f'./commit-param {pp_path} {int_bin_path} {commitment_path} {w_out.shape[0]} {w_out.shape[1]}')
            else:
                os.system(f'./commit-param {pp_path} {int_bin_path} {commitment_path} {w_out.shape[0]} 1')

    print('Conversion complete. Output in', outdir)
