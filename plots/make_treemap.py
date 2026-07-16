#!/usr/bin/env python3

import os
from collections import Counter
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

import benchmark_plots as bp

FONT = "DejaVu Sans"

BG = "#EFF7EA"
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

# Shorter display labels for long benchmark names.
# Otherwise, they won't fit into the treemap.
SHORT_LABEL = {
    "NAS Parallel Benchmarks": "NAS",
    "Python perf. benchmark": "pyperformance",
    "Yahoo! Streaming Benchmark": "Yahoo Str.\nBenchmark",
    "OpenFaaS Function Store": "OpenFaaS\nStore",
    "TensorFlow Federated": "TFF",
    "Online Boutique": "Online Boutique",
    "Serverless Examples": "Serverless\nExamples",
    "Predicting the Costs of Serverless Workflows": "Predicting Costs of\nServerless Workflows",
}

# Origin domain of each imported benchmark.
# Manually map benchmarks to domains
OTHER_DOMAIN = {
    "DeathStarBench": "Microservices",
    "Online Boutique": "Microservices",
    "TPC": "Databases & Analytics",
    "HiBench": "Databases & Analytics",
    "NEXMark": "Databases & Analytics",
    "TeraSort": "Databases & Analytics",
    "Montage": "HPC & Scientific",
    "ExaMol": "HPC & Scientific",
    "LNNI": "HPC & Scientific",
    "BERT": "ML Models & Suites",
    "imgclsmob": "ML Models & Suites",
    "NASBench": "ML Models & Suites",
    "OpenAI Gym": "ML Models & Suites",
    "CodeSearchNet": "ML Models & Suites",
    "DISB": "ML Models & Suites",
    "SPECint": "CPU/GPU Benchmarks",
    "SPECjbb": "CPU/GPU Benchmarks",
    "PolyBench": "CPU/GPU Benchmarks",
    "stress": "CPU/GPU Benchmarks",
    "Python perf. benchmark": "CPU/GPU Benchmarks",
    "PageRank": "Graph Processing",
    "FStartBench": "Serverless Tools",
    "XFBench": "Serverless Tools",
    "LambdaML": "Serverless Tools",
    "FaaSTest": "Serverless Tools",
    "AWS Samples": "Serverless Tools",
    "Sprocket": "Serverless Tools",
    "Corral": "Serverless Tools",
    "OpenFaaS Function Store": "Serverless Tools",
    "Serverless Examples": "Serverless Tools",
    "PARSEC": "CPU/GPU Benchmarks",
    "Rodinia": "CPU/GPU Benchmarks",
    "NAS Parallel Benchmarks": "HPC & Scientific",
    "Task Bench": "HPC & Scientific",
    "Pegasus": "HPC & Scientific",
    "WfCommons": "HPC & Scientific",
    "YCSB": "Databases & Analytics",
    "db_bench": "Databases & Analytics",
    "sysbench": "Databases & Analytics",
    "MLPerf": "ML Models & Suites",
    "FedScale": "ML Models & Suites",
    "LEAF": "ML Models & Suites",
    "TensorFlow Federated": "ML Models & Suites",
    "RIoTBench": "Streaming / IoT",
    "Yahoo! Streaming Benchmark": "Streaming / IoT",
    "HelloRetail": "Microservices",
    "Prior papers": "Prior papers",
}

# 'Other papers' category for the treemap: workloads drawn from individual
# papers (replaces the former 'Prior papers' aggregate).
#
# Manually calculated; cross verified with Claude Fable.

OTHER_PAPERS = {
    "FaaSProfiler": 7,
    "ExCamera": 3,
    "Sprocket": 2,
    "Pocket": 2,
    "Cirrus": 2,
    "Predicting the Costs of Serverless Workflows": 2,
    "gg": 1,
    "NumpyWren": 1,
    "OFC": 1,
    "ORION": 1,
    "Beldi": 1,
    "Owl": 1,
    "Llama": 1,
    "profaastinate": 1,
    "wisefuse": 1,
    "Faster/cheaper on harvested resources": 1,
    "Faastlane: Accelerating Function-as-a-Service Workflow": 1,
    "Implications of Prog. Lang. Selection": 1,
    "Characterizing Commodity Serverless Computing Platforms (slsbench)": 1,
    "From warm to hot starts": 1,
    "Using a microbenchmark": 1,
    "Modeling and Optimization of Performance and Cost of Serverless Applications.": 1,
    "Predicting Performance and Cost of Serverless Computing Functions with SAAF": 1,
    "Serverless performance modeling with cpu time accounting and the serverless application analytics framework": 1,
    "Function memory optimization for heterogeneous serverless platforms with cpu time accounting": 1,
    "estimating function completion times (2021)": 1,
    "event-driven serverless (Moina-Rivera)": 1,
    "A not so cold (2020)": 1,
    "descriptive and predictive analysis of aggregating functions in serverless clouds": 1,
    "serverless computing: An investigation of factors influencing microservice performance (2018)": 1,
}


def plot_treemap(
    rel,
    outdir,
    show_title=True,
    merge_below=3,
    collapse_uniform=("ML Models & Suites", "HPC & Scientific"),
):
    import squarify

    counts = Counter()
    for names in rel["Other_names"]:
        for n in names:
            counts[n] += 1
    items = pd.DataFrame(
        [(n, c, OTHER_DOMAIN.get(n, "Unclassified")) for n, c in counts.items()],
        columns=["name", "count", "domain"],
    )
    unclassified = items[items["domain"] == "Unclassified"]
    if len(unclassified):
        print(
            "WARNING: unclassified imported benchmarks:", unclassified["name"].tolist()
        )
        raise RuntimeError(
            "Unclassified imported benchmarks found; please update OTHER_DOMAIN."
        )

    # Drop non-benchmark categories (e.g. 'Prior papers') from this figure.
    # we will explicitly introduce manually counted paper usage
    exclude_domains = ("Prior papers",)
    items = items[~items["domain"].isin(exclude_domains)].copy()

    # we manually counted serverless papers used.
    # we fold entries below 2 and also support "Other" category (used previously)
    existing_names = set(items["name"])
    op_rows, folded = [], {}
    for name, cnt in OTHER_PAPERS.items():
        if name in existing_names:
            print(
                f"WARNING: '{name}' in OTHER_PAPERS also appears as an "
                f"imported benchmark elsewhere in the treemap (double-counted)."
            )
        if name == "Other":
            folded[name] = folded.get(name, 0) + cnt  # explicit 'Other'
        elif cnt <= 1:
            folded["__fold__"] = folded.get("__fold__", 0) + cnt
        else:
            op_rows.append((name, cnt, "Serverless Papers"))
    combined = folded.get("__fold__", 0) + folded.get("Other", 0)
    if combined:
        op_rows.append(("Other", combined, "Serverless Papers"))
    items = pd.concat(
        [items, pd.DataFrame(op_rows, columns=["name", "count", "domain"])],
        ignore_index=True,
    )

    # Within selected domains where every benchmark is used once, collapse the
    # individual tiles into a single tile (label shows # of distinct benchmarks).
    for dom in collapse_uniform:
        sub = items[items["domain"] == dom]
        if len(sub) > 1 and (sub["count"] == 1).all():
            n = len(sub)
            # everything except collaped domains
            items = items[items["domain"] != dom].copy()
            items = pd.concat(
                [
                    items,
                    pd.DataFrame(
                        [{"name": f"{n} benchmarks, 1 each", "count": n, "domain": dom}]
                    ),
                ],
                ignore_index=True,
            )

    # Collapse domains whose total use count is < merge_below into 'Other domains'.
    domain_totals = items.groupby("domain")["count"].sum()
    small = domain_totals[domain_totals < merge_below].index.tolist()
    if small:
        items.loc[items["domain"].isin(small), "domain"] = "Other Domains"
        print("Merged into 'Other domains':", small)

    domain_order = items.groupby("domain")["count"].sum().sort_values(ascending=False)
    palette = {
        "Microservices": "#d1495b",
        "Databases & Analytics": "#2e6f95",
        "ML Models & Suites": "#52854c",
        "HPC & Scientific": "#b07a30",
        "CPU/GPU Benchmarks": "#7b6d8d",
        "Serverless Tools": "#5fa8a0",
        "Graph Processing": "#c47ac0",
        "Streaming / IoT": "#3d8b9c",
        "Other Domains": "#9d8f7a",
        "Serverless Papers": "#8d8d8d",
    }

    # default font too large, spills over.
    FONT_SIZE_OVERRIDE = {
        "Yahoo Streaming\n Benchmark": 4.5,
        "Python perf. benchmark": 4.8,
    }

    # Nested layout: outer rectangles per domain, inner per benchmark.
    fig, ax = plt.subplots(figsize=(9, 4.0))
    norm = squarify.normalize_sizes(domain_order.values, 100, 100)
    outer = squarify.squarify(norm, 0, 0, 100, 100)
    for rect, (domain, total) in zip(outer, domain_order.items()):
        sub = items[items["domain"] == domain].sort_values("count", ascending=False)
        pad = 0.6
        header_h = min(5.5, rect["dy"] * 0.28)  # band for the domain label
        x, y = rect["x"] + pad, rect["y"] + pad
        dx = max(rect["dx"] - 2 * pad, 0.1)
        dy = max(rect["dy"] - 2 * pad - header_h, 0.1)
        inner_norm = squarify.normalize_sizes(sub["count"].values, dx, dy)
        inner = squarify.squarify(inner_norm, x, y, dx, dy)
        for irect, (_, row) in zip(inner, sub.iterrows()):
            ax.add_patch(
                plt.Rectangle(
                    (irect["x"], irect["y"]),
                    irect["dx"],
                    irect["dy"],
                    facecolor=palette[domain],
                    edgecolor="white",
                    linewidth=1.2,
                    alpha=0.55 + 0.45 * row["count"] / sub["count"].max(),
                )
            )
            label = SHORT_LABEL.get(row["name"], row["name"])
            iw, ih = irect["dx"], irect["dy"]
            if iw * ih > 14 and iw > 3.5:
                # font scaled to area, then capped so the label fits the width
                # respect overrides
                fs = FONT_SIZE_OVERRIDE.get(row["name"], min(9.0, 4.5 + iw * ih / 70))

                # ~0.62 coord-units per character at fs=10 (empirical)
                char_w = 0.062 * fs
                max_chars = max(4, int(iw / char_w))
                txt = (
                    label
                    if len(label) <= max_chars
                    else label[: max_chars - 1] + "\u2026"
                )

                ax.text(
                    irect["x"] + iw / 2,
                    irect["y"] + ih / 2,
                    f"{txt}\n({row['count']})",
                    ha="center",
                    va="center",
                    fontsize=fs,
                    color="white",
                    weight="bold",
                )
        ax.add_patch(
            plt.Rectangle(
                (rect["x"], rect["y"]),
                rect["dx"],
                rect["dy"],
                fill=False,
                edgecolor="black",
                linewidth=1.8,
            )
        )
        ax.text(
            rect["x"] + 1.2,
            rect["y"] + rect["dy"] - 1.2,
            f"{domain} ({total})",
            ha="left",
            va="top",
            fontsize=8,
            weight="bold",
            # bbox=dict(facecolor="#EFF7EA", alpha=0.85, edgecolor="none", pad=1.5),
        )
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    if show_title:
        ax.set_title(
            "Benchmarks imported from other domains "
            f"({int(items['count'].sum())} uses across "
            f"{int(rel['Other_names'].str.len().gt(0).sum())} papers)",
            fontsize=11,
        )
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(
            outdir / f"fig_treemap_imported.{ext}", bbox_inches="tight", dpi=200
        )
    plt.close(fig)
    return items


rel = bp.load_and_clean(os.path.join(os.path.pardir, "papers_study.xlsx"))
plot_treemap(rel, Path("figures"), show_title=False)
