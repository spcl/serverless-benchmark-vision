#!/usr/bin/env python3

from pathlib import Path
import os
import pandas as pd

import matplotlib.pyplot as plt

import benchmark_plots as bp

# uses the upsetplot library that is no loner maintained
#
# two issues:
# - does not work with pandas, uv pip install pandas==2.3.3
# - does not work with numpy 2.4, there's a fork that fixes that
# uv pip install upsetplot@git+https://github.com/durr1602/UpSetPlot.git@7fa2661
#
# from: https://github.com/jnothman/UpSetPlot/issues/301
#
# However, I don't seen an option to properly respect counts
# when we don't plot values < 1

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

    import upsetplot as up
    from matplotlib import pyplot

    up.plot(combos, show_counts=True, sort_by="cardinality", sort_categories_by=None)

    ext = "pdf"
    pyplot.savefig(f"fig_upset.{ext}", bbox_inches="tight", dpi=200)
    ext = "png"
    pyplot.savefig(f"fig_upset.{ext}", bbox_inches="tight", dpi=300)


rel = bp.load_and_clean(os.path.join("papers_study.xlsx"))

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

plot_upset(rel, show_title=False)
