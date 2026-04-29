import os, sys
import argparse
import torch
import numpy as np

parser = argparse.ArgumentParser(description='OPT Self-Attention test')
parser.add_argument('layer', type=int, help='Layer index')
parser.add_argument('seq_len', type=int, help='Sequence length')
parser.add_argument('--input_file', required=True, type=str, help='The input int32 bin file')
parser.add_argument('--output_file', default='opt-self-attn-output.bin', type=str, help='Output file')

from transformers import AutoModelForCausalLM
from fileio_utils import *

VALUE_LOGSF = 16
ACCU_LOGSF = 20

if __name__ == '__main__':
    args = parser.parse_args()
    repo_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(repo_root)

    compilation_error = os.system('make self-attn')
    if compilation_error:
        print("Error compiling self-attn")
        # continue in case binary already exists

    model = AutoModelForCausalLM.from_pretrained('/home/daqi/opt', local_files_only=True)
    layer = model.model.decoder.layers[args.layer]
    embed_dim = layer.self_attn.q_proj.in_features

    workdir = f'./zkllm-workdir/OPT-125m'
    layer_prefix = f'layer-{args.layer}'

    # call linear proof
    cmd = f'./self-attn linear {args.input_file} {args.seq_len} {embed_dim} {workdir} {layer_prefix} {args.output_file}'
    print('RUN:', cmd)
    ret = os.system(cmd)
    print('linear exit code', ret)

    # attempt to run attn step
    cmd2 = f'./self-attn attn {args.input_file} {args.seq_len} {embed_dim} {workdir} {layer_prefix} {args.output_file}'
    print('RUN:', cmd2)
    ret2 = os.system(cmd2)
    print('attn exit code', ret2)
