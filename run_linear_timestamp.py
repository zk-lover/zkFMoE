#!/usr/bin/env python3
import subprocess, time, sys
cmd=['./self-attn','linear','opt-input-int-padded.bin','4','1024','./zkllm-workdir/OPT-125m','layer-0','opt-linear-out.bin']
print('CMD:', ' '.join(cmd), file=sys.stderr)
with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as p:
    lines=[]
    start=time.time()
    for line in p.stdout:
        t=time.time()
        s=line.rstrip('\n')
        print(f"{t:.6f} {s}")
        lines.append((t,s))
    ret=p.wait()
    end=time.time()
    print('RET',ret,file=sys.stderr)
    with open('linear_timestamped.txt','w') as f:
        f.write(f"start\t{start:.6f}\n")
        f.write(f"end\t{end:.6f}\n")
        f.write(f"total\t{end-start:.6f}\n")
        for t,s in lines:
            f.write(f"{t:.6f}\t{s}\n")
    print('WROTE linear_timestamped.txt',file=sys.stderr)
    print('TOTAL', end-start, file=sys.stderr)
