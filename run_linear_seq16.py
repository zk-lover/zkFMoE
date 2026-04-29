#!/usr/bin/env python3
import subprocess, time, sys
cmd=['./self-attn','linear','opt-input-int-padded-seq16.bin','16','1024','./zkllm-workdir/OPT-125m','layer-0','opt-linear-out-seq16.bin']
print('CMD:', ' '.join(cmd))
with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as p:
    lines=[]
    start=time.time()
    for line in p.stdout:
        t=time.time(); print(f"{t:.6f} {line.rstrip()}")
        lines.append((t,line))
    ret=p.wait(); end=time.time()
    print('RET',ret)
    with open('linear_seq16_timestamped.txt','w') as f:
        f.write(f'start\t{start:.6f}\n')
        f.write(f'end\t{end:.6f}\n')
        f.write(f'total\t{end-start:.6f}\n')
        for t,l in lines:
            f.write(f"{t:.6f}\t{l}")
    print('WROTE linear_seq16_timestamped.txt')
