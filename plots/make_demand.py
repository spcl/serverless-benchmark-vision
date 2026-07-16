#!/usr/bin/env python3

import os
from collections import Counter

import matplotlib.pyplot as plt

import benchmark_plots as bp

# ---------------------------------------------------------------- font
FONT = "DejaVu Sans"

BG = "#FFFFFF"
plt.rcParams.update(
    {
        "font.family": FONT,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "savefig.facecolor": BG,
        "savefig.edgecolor": BG,
    }
)

# ---------------------------------------------------------------- data
rel = bp.load_and_clean(os.path.join(os.path.pardir, "papers_study.xlsx"))

# Demand = workload domains of CUSTOM papers only (from the 'Workload' column).
# Imported/'Other (list)' benchmarks are intentionally excluded here.
demand = Counter()
for tags in rel.loc[rel["Custom"] == 1, "Workload_tags"]:
    for t in tags:
        demand[t] += 1

# Fold the long tail of small categories into a single 'Other' bar.
# Everything with count <= TAIL_MAX is aggregated (mirrors the earlier
# "everything past Workflow" cut, but robust to which labels are present).
TAIL_MAX = 3
ordered = sorted(demand, key=lambda d: -demand[d])
named = [d for d in ordered if demand[d] > TAIL_MAX]
tail = [d for d in ordered if demand[d] <= TAIL_MAX]
rows = list(named)
values = [demand[d] for d in named]
if tail:
    rows.append("Other")
    values.append(sum(demand[d] for d in tail))
    print("Folded into 'Other':", {d: demand[d] for d in tail})

fig, ax = plt.subplots(figsize=(6.4, 0.40 * len(rows) + 1.5))
ypos = list(range(len(rows)))[::-1]
colors = ["#33518e"] * (len(rows) - (1 if tail else 0)) + (["#9aa6c4"] if tail else [])
ax.barh(ypos, values, color=colors, height=0.68)
for y, v in zip(ypos, values):
    ax.text(
        v + max(values) * 0.012, y, str(v), va="center", fontsize=10, fontfamily=FONT
    )

ax.set_yticks(ypos)
ax.set_yticklabels(rows, fontsize=11)
ax.set_xlabel("")  # y-axis description dropped
ax.set_xlim(0, max(values) * 1.1)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(length=0)

fig.tight_layout()
ext = "png"
fig.savefig(f"figures/fig_demand.{ext}", bbox_inches="tight", dpi=300)
ext = "pdf"
fig.savefig(f"figures/fig_demand.{ext}", bbox_inches="tight", dpi=300)
