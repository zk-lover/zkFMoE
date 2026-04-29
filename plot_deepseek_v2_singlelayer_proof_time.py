# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import numpy as np

plt.rcParams["axes.unicode_minus"] = False
title_font = fm.FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", size=14)

# Data for DeepSeek-V2 (236B), single layer
seq_lens = [256, 512, 1024, 2048, 4096]
attn_proving_time = [0.467203, 0.830725, 2.225209, 6.451958, 22.596014]
moe_proving_time = [0.073674, 0.107342, 0.174611, 0.291893, 0.547600]

x = np.arange(len(seq_lens))
width = 0.32

# Figure 1: attn only, keep the original axis specification
fig, ax = plt.subplots(figsize=(10 * 2 / 3, 6))

ax.bar(x, attn_proving_time, width, color="#4C78A8")

ax.set_title("single layer attention proof time (s)", fontproperties=title_font, fontsize=20)
ax.set_xlabel("sequence length",fontsize=18)
ax.set_ylabel("proof time",fontsize=18)
ax.tick_params(axis='both', labelsize=16)
ax.set_xticks(x)
ax.set_xticklabels([str(v) for v in seq_lens])
ax.set_ylim(0, 25)
ax.set_yticks(np.arange(0, 25.1, 2.5))
ax.grid(axis="y", linestyle="--", alpha=0.35)

plt.tight_layout()
attn_out_file = "deepseek_v2_singlelayer_attn_proof_time_bar.png"
plt.savefig(attn_out_file, dpi=200)
print(f"Saved figure to: {attn_out_file}")
plt.close(fig)

# Figure 2: MoE only, adjust y-axis for smaller values
fig, ax = plt.subplots(figsize=(10 * 2 / 3, 6))

ax.bar(x, moe_proving_time, width, color="#F58518")

ax.set_title("single layer Top-k MoE proof time (s)", fontproperties=title_font, fontsize=20)
ax.set_xlabel("sequence length", fontsize=18)
ax.set_ylabel("proof time", fontsize=18)
ax.tick_params(axis='both', labelsize=16)
ax.set_xticks(x)
ax.set_xticklabels([str(v) for v in seq_lens])
ax.set_ylim(0, 0.6)
ax.set_yticks(np.arange(0, 0.61, 0.1))
ax.grid(axis="y", linestyle="--", alpha=0.35)

plt.tight_layout()
moe_out_file = "deepseek_v2_singlelayer_moe_proof_time_bar.png"
plt.savefig(moe_out_file, dpi=200)
print(f"Saved figure to: {moe_out_file}")
