#!/usr/bin/env python3

from pathlib import Path
import os
import pandas as pd

import matplotlib.pyplot as plt

import benchmark_plots as bp

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


def plot_upset(
    rel,
    outdir,
    min_subset_size=2,
    show_title=True,
    exclude=("Custom", "Imported benchmark"),
):
    """Self-contained UpSet plot (upsetplot 0.9 is incompatible with pandas 3)."""
    # https://github.com/jnothman/UpSetPlot/issues/303
    ann = rel[rel["any_annotation"]].copy()
    ind = pd.DataFrame(
        {
            "SeBS": ann["SeBS"] == 1,
            "FunctionBench": ann["FunctionBench"] == 1,
            "ServerlessBench": ann["ServerlessBench"] == 1,
            "Other suite": ann[["vSwarm", "Triggerbench", "Servibench", "FaasDom"]].sum(
                axis=1
            )
            > 0,
            "Microbenchmark": ann["Microbenchmark"] == 1,
            "Custom": ann["Custom"] == 1,
            "Imported benchmark": ann["Other_names"].str.len() > 0,
        }
    )
    # now we have binary matrix - for each row, True if a given benchmark was used.
    # remove custom/imported - the plot is too busy
    # We show custom and imported on different benchmarks
    ind = ind.drop(columns=[c for c in exclude if c in ind.columns])
    # drop papers that do not use any of the selected benchmark (we dropped columns)
    ind = ind[ind.any(axis=1)]  # keep papers with ≥1 kept cat

    # count sums per column - total number of usage per benchmark
    totals = ind.sum().sort_values(ascending=False)
    print("count sums per column - total number of usage per benchmark")
    print(totals)

    # we should get now all benchmarks sorted by size - this is just names
    cats = totals.index.tolist()  # rows sorted by set size

    # group by each category and count number of combinations
    # this should produce one group for each combination of benchmarks
    # size will count the number of occurences
    combos = ind.groupby(cats).size().sort_values(ascending=False)
    # cut combinations with only one occurence - filter threshold
    combos = combos[combos >= min_subset_size]
    n_shown = int(combos.sum())

    nrows, ncols = len(cats), len(combos)
    fig = plt.figure(figsize=(0.42 * ncols + 2.6, 0.34 * nrows + 2.6))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.1, 0.34 * ncols],
        height_ratios=[2.0, 0.34 * nrows],
        hspace=0.06,
        wspace=0.04,
    )
    # barplot with occurences
    ax_bar = fig.add_subplot(gs[0, 1])
    # the matrix with all combinations
    ax_mat = fig.add_subplot(gs[1, 1], sharex=ax_bar)
    # set of each size included
    ax_set = fig.add_subplot(gs[1, 0], sharey=ax_mat)

    # intersection-size bars
    # show the value of each combination
    xs = range(ncols)
    ax_bar.bar(xs, combos.values, color="#33518e", width=0.6)
    for x, v in zip(xs, combos.values):
        ax_bar.text(
            x,
            v + max(combos.values) * 0.02,
            str(v),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax_bar.set_ylabel("Papers", fontsize=9)
    ax_bar.spines[["top", "right"]].set_visible(False)
    ax_bar.tick_params(labelbottom=False, bottom=False)
    ax_bar.set_ylim(0, max(combos.values) * 1.12)

    # membership matrix
    for x, key in enumerate(combos.index):
        members = dict(zip(combos.index.names, key))
        active = [r for r, c in enumerate(cats) if members[c]]
        ax_mat.scatter(
            [x] * nrows,
            range(nrows),
            s=58,
            color=["#33518e" if r in active else "#dddddd" for r in range(nrows)],
            zorder=3,
        )
        if len(active) > 1:
            ax_mat.plot(
                [x, x], [min(active), max(active)], color="#33518e", lw=2, zorder=2
            )
    ax_mat.set_yticks(range(nrows))
    ax_mat.set_yticklabels(cats, fontsize=9)
    ax_mat.yaxis.tick_right()
    ax_mat.set_xlim(-0.6, ncols - 0.4)
    ax_mat.set_ylim(nrows - 0.4, -0.6)  # largest set on top
    ax_mat.tick_params(left=False, right=False, bottom=False, labelbottom=False)
    for s in ax_mat.spines.values():
        s.set_visible(False)
    bg = plt.rcParams.get("axes.facecolor", "#FFFFFF")
    stripe = {"#EFF7EA": "#e1efd6"}.get(str(bg).upper(), "#f0f0f0")
    for r in range(nrows):
        ax_mat.axhspan(r - 0.5, r + 0.5, color=stripe if r % 2 else bg, zorder=0)

    # set-size bars (left, mirrored)
    ax_set.barh(range(nrows), [totals[c] for c in cats], color="#8d8d8d", height=0.55)
    for r, c in enumerate(cats):
        ax_set.text(
            totals[c] + max(totals) * 0.03,
            r,
            str(int(totals[c])),
            va="center",
            ha="right",
            fontsize=8,
        )
    ax_set.set_xlim(max(totals) * 1.22, 0)  # mirrored
    ax_set.set_xlabel("Total occurences", fontsize=9)
    ax_set.spines[["top", "left", "right"]].set_visible(False)
    ax_set.tick_params(left=False, labelleft=False)

    if show_title:
        fig.suptitle(
            f"Benchmark co-usage across {len(ind)} annotated papers "
            f"({n_shown} in intersections with ≥{min_subset_size} papers)",
            fontsize=10,
            y=0.98,
        )
    ext = "pdf"
    fig.savefig(outdir / f"fig_upset.{ext}", bbox_inches="tight", dpi=200)
    ext = "png"
    fig.savefig(outdir / f"fig_upset.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


rel = bp.load_and_clean(os.path.join(os.path.pardir, "papers_study.xlsx"))

# print statistics for analysis

for bench in [
    "SeBS",
    "FunctionBench",
    "ServerlessBench",
    "vSwarm",
    "Triggerbench",
    "Servibench",
    "FaasDom",
    "Microbenchmark",
    "Custom",
]:

    print(bench, (rel[bench] == 1).sum())

plot_upset(rel, Path("figures"), show_title=False)
