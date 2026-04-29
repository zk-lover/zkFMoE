import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

models = [
    "opt-6.7B",
    "opt-13B",
    "LLaMA-2-13B",
    "Qwen-32B",
    "DeepSeek-V2",
    "DeepSeek-V3.2",
]

this_work = np.array([281.71, 522.38, 514.71, 793.76, 1551.06, 2210.62], dtype=float)
zkllm = np.array([548.0, 713.0, 803.0, np.nan, np.nan, np.nan], dtype=float)

x = np.arange(len(models))
width = 0.35
title_font = fm.FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", size=14)

fig, ax = plt.subplots(figsize=(11, 6.5))

bars_this = ax.bar(x - width / 2, this_work, width, label="zkFMoE", color="#4C78A8")
bars_zk = ax.bar(x + width / 2, zkllm, width, label="zkLLM", color="#F58518")

ax.set_title("Proof time comparison", fontproperties=title_font)
ax.set_xlabel("Model")
ax.set_ylabel("Prover time (s)")
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=20, ha="right")
ax.grid(axis="y", linestyle="--", alpha=0.3)
ax.legend(frameon=False)

# Label each visible bar with its value.
for bar in bars_this:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 18, f"{h:.2f}", ha="center", va="bottom", fontsize=9)

for bar, val in zip(bars_zk, zkllm):
    if np.isnan(val):
        continue
    h = bar.get_height()
    text = f"{h:.0f}" if float(h).is_integer() else f"{h:.2f}"
    ax.text(bar.get_x() + bar.get_width() / 2, h + 18, text, ha="center", va="bottom", fontsize=9)

# Mark missing zkLLM results as N/A.
for i, val in enumerate(zkllm):
    if np.isnan(val):
        ax.text(x[i] + width / 2, 40, "N/A", ha="center", va="bottom", fontsize=10, color="#C44E52")

ax.set_ylim(0, 2400)

plt.tight_layout()
out_file = "/home/daqi/zkllm-ccs2024/prover_time_comparison_thiswork_vs_zkllm.png"
plt.savefig(out_file, dpi=220)
print(f"Saved figure to: {out_file}")
