#!/usr/bin/env python3
"""
Prototype: ZK-Online-Softmax (block-wise online softmax compatible with zkllm pipeline).

Usage:
  python tools/zkonline_softmax.py --q q.npy --k k.npy --out attn_int.npy --block_size 64 --scale 4096

This script performs block-wise (online) softmax over Q @ K^T with numeric-stable merging
and writes a quantized int32 attention matrix to `--out`.
"""
import argparse
import numpy as np
import os


def load_array(path):
    if path.endswith('.npy'):
        return np.load(path)
    raise RuntimeError('Only .npy input is supported in prototype')


def online_block_softmax(Q, K, block_size=64):
    # Q: (T, D), K: (S, D) -> output (T, S)
    T, D = Q.shape
    S = K.shape[0]
    out_blocks = []
    # running max and sum per query
    M = np.full((T,), -np.inf, dtype=np.float64)
    Ssum = np.zeros((T,), dtype=np.float64)

    for start in range(0, S, block_size):
        Kb = K[start:start+block_size]  # (B, D)
        # scores: (T, B)
        scores = Q.dot(Kb.T) / np.sqrt(D)
        mb = np.max(scores, axis=1)
        eb = np.exp(scores - mb[:, None])  # (T, B)
        sumb = eb.sum(axis=1)

        # merge previous and block-wise maxima/sums
        M_new = np.maximum(M, mb)
        # adjust previous sum to new max
        Ssum = Ssum * np.exp(M - M_new) + sumb * np.exp(mb - M_new)

        # compute normalized block outputs relative to new M
        # out_block = exp(scores - M_new[:,None]) / Ssum[:,None]
        # = eb * exp(mb - M_new) / Ssum[:,None]
        factor = np.exp(mb - M_new)
        out_block = (eb * factor[:, None]) / Ssum[:, None]

        out_blocks.append((start, out_block.astype(np.float32)))

        M = M_new

    # assemble full attention matrix
    attn = np.zeros((T, S), dtype=np.float32)
    for start, ob in out_blocks:
        B = ob.shape[1]
        attn[:, start:start+B] = ob
    return attn


def quantize_and_save(arr, out_path, scale=4096):
    # quantize to int32 with simple rounding
    iarr = np.rint(arr * scale).astype(np.int32)
    np.save(out_path, iarr)
    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--q', required=True)
    p.add_argument('--k', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--block_size', type=int, default=64)
    p.add_argument('--scale', type=float, default=4096.0)
    args = p.parse_args()

    Q = load_array(args.q)
    K = load_array(args.k)
    attn = online_block_softmax(Q, K, block_size=args.block_size)
    out = quantize_and_save(attn, args.out, scale=args.scale)
    print('Wrote quantized attention to', out)


if __name__ == '__main__':
    main()
