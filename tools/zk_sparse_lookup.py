#!/usr/bin/env python3
"""
Prototype: Sparse expert lookup argument helper.

Usage:
  python tools/zk_sparse_lookup.py --logits logits.npy --k 1 --out mask.npy

Given per-token expert logits (T, E), produce a Top-k mask (T, E) and save as int8 numpy.
This prototype supports Top-1 by default and writes selected expert indices as well.
"""
import argparse
import numpy as np


def load_array(path):
    if path.endswith('.npy'):
        return np.load(path)
    raise RuntimeError('Only .npy input is supported in prototype')


def topk_mask(logits, k=1):
    # logits: (T, E)
    indices = np.argpartition(-logits, range(k), axis=1)[:, :k]
    T, E = logits.shape
    mask = np.zeros((T, E), dtype=np.int8)
    for i in range(T):
        mask[i, indices[i]] = 1
    return mask, indices


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--logits', required=True)
    p.add_argument('--k', type=int, default=1)
    p.add_argument('--out', required=True)
    args = p.parse_args()

    logits = load_array(args.logits)
    mask, idx = topk_mask(logits, k=args.k)
    np.save(args.out, mask)
    idx_out = args.out.replace('.npy', '.idx.npy')
    np.save(idx_out, idx)
    print('Wrote mask to', args.out, 'and indices to', idx_out)


if __name__ == '__main__':
    main()
