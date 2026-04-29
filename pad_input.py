#!/usr/bin/env python3
import numpy as np
import sys

inpath='opt-input-int-padded.bin'
outpath='opt-input-int-padded-seq16.bin'
old_seq=4
new_seq=16
embed=1024

arr = np.fromfile(inpath, dtype=np.int32)
if arr.size != old_seq*embed:
    print('Unexpected input size', arr.size, 'expected', old_seq*embed)
    sys.exit(1)

arr = arr.reshape((old_seq, embed))
pad_rows = new_seq - old_seq
# repeat the existing rows to fill (keeps distribution similar)
repeats = (pad_rows + old_seq - 1) // old_seq
pad = np.tile(arr, (repeats, 1))[:pad_rows]
new = np.vstack([arr, pad])
new.astype(np.int32).tofile(outpath)
print('Wrote', outpath, 'with shape', new.shape)
