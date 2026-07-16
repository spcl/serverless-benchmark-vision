#!/usr/bin/env python3

import re
import pandas as pd

SUITES = [
    "SeBS",
    "FunctionBench",
    "ServerlessBench",
    "vSwarm",
    "Triggerbench",
    "Servibench",
    "FaasDom",
]

# Canonical workload-domain labels for the 'Workload' column (custom workloads).
WORKLOAD_ALIASES = {
    "ml": "ML",
    "data analytics": "Data Analytics",
    "hpc": "HPC & Scientific",
    "scientific": "HPC & Scientific",
    "edge": "Edge",
    "video": "Video & Multimedia",
    "vid": "Video & Multimedia",
    "multimedia": "Video & Multimedia",
    "streaming": "Streaming",
    "stream processing": "Streaming",
    "graph": "Graph Processing",
    "graph processing": "Graph Processing",
    "webapp": "Web Application",
    "data movement": "Data Movement",
    "data processing": "Data Analytics",
    "text analysis": "Data Analytics",
    "text processing": "Data Analytics",
    "event processing": "Streaming",
    "database": "Database",
    "data storage": "Database",
    "nfv": "Network (NFV)",
    "fpga": "Accelerator (FPGA)",
    "cpu": "Generic CPU kernels",
    "encryption": "Generic CPU kernels",
    "image processing": "Video & Multimedia",
    "i/o": "I/O",
    "synthetic": "Synthetic & Underspecified",
    "underspecified": "Synthetic & Underspecified",
    "unspecified": "Synthetic & Underspecified",
}

# Canonical names for entries in 'Other (list)' (typo/case variants).
#
# We also collapse different benchmarks and their variants into a single representation.
# Different TPC variants -> TPC
# Different SPEC benchmarks -> SPEC
OTHER_ALIASES = {
    "deathstarbench": "DeathStarBench",
    "deatstarbench": "DeathStarBench",
    "tpc-ds": "TPC",
    "tpc-h": "TPC",
    "tpc-w": "TPC",
    "tpc": "TPC",
    "hibench": "HiBench",
    "nexmark": "NEXMark",
    "terasort": "TeraSort",
    "specint": "SPEC CPU",
    "specjbb": "SPECjbb",
    "specjbb2015": "SPECjbb",
    "polybench": "PolyBench",
    "polybench/c": "PolyBench",
    "python performance benchmark": "Python perf. benchmark",
    "montage": "Montage",
    "examol": "ExaMol",
    "lnni": "LNNI",
    "hpc workflows": "HPC workflows",
    "llama": "Llama",
    "bert": "BERT",
    "imgclsmob": "imgclsmob",
    "nasbench": "NASBench",
    "lambdaml": "LambdaML",
    "openai gym": "OpenAI Gym",
    "codesearchnet": "CodeSearchNet",
    "disb (subset)": "DISB",
    "online boutique": "Online Boutique",
    "fstartbench": "FStartBench",
    "faastest": "FaaSTest",
    "aws samples": "AWS Samples",
    "sprocket": "Sprocket",
    "corral": "Corral",
    "pagerank": "PageRank",
    "papers": "Prior papers",
    "disb": "DISB",
    "pyperformance": "Python perf. benchmark",
    "parsec": "PARSEC",
    "rodinia": "Rodinia",
    "nas": "NAS Parallel Benchmarks",
    "ycsb": "YCSB",
    "db_bench": "db_bench",
    "sysbench": "sysbench",
    "mlperf": "MLPerf",
    "fedscale": "FedScale",
    "leaf": "LEAF",
    "tff": "TensorFlow Federated",
    "riotbench": "RIoTBench",
    "yahoo streaming benchmark": "Yahoo! Streaming Benchmark",
    "helloretail": "HelloRetail",
    "pegasus": "Pegasus",
    "wfcommons": "WfCommons",
    "openfaas function store": "OpenFaaS Function Store",
    "serverless examples": "Serverless Examples",
    "stress": "stress",
    "task bench": "Task Bench",
    "xfbench": "XFBench",
}


# Demand-coverage mapping: for each demanded domain, whether it is covered by
# (1) existing dedicated serverless suites, (2) the proposed Baseline tier,
# (3) the proposed Extended tier (Table 2 of the paper).
# Values: True = covered, "partial" = partially covered, False = not covered.
COVERAGE = {
    # domain:               (existing suites, Baseline tier, Extended tier)
    "ML": ("partial", True, True),  # SeBS inference / III B+E
    "Data analytics": ("partial", True, True),  # II: MapReduce, ETL, SQL
    "HPC / scientific": (False, False, True),  # VII
    "Edge": (False, False, False),  # absent from Table 2
    "Video / multimedia": ("partial", True, False),  # vSwarm video / IV media
    "Streaming": (False, False, True),  # II stream processing
    "Graph processing": ("partial", True, False),  # SeBS BFS / I graph
    "Workflow": ("partial", True, True),  # SeBS-Flow / II, V
    "Web application": (True, True, False),  # I HTML, V CRUD backend
    "Data movement": ("partial", True, False),  # II ETL, IV media
    "Microservices": (False, True, True),  # IV
    "CPU/GPU Benchmarks": (True, True, False),  # I core FaaS kernels
    "Distributed builds": (False, False, True),  # VI
}

# Imported-benchmark domains folded into demand-domain labels.
IMPORT_TO_DEMAND = {
    "ML models & Suites": "ML",
    "Databases & Analytics": "Data analytics",
    "Classic CPU suites": "Generic CPU kernels",
    "HPC / scientific": "HPC / scientific",
    "Microservices": "Microservices",
    "Graph processing": "Graph processing",
    "Streaming / IoT": "Streaming",
}


def split_tags(value, seps=r"[,;/]"):
    """Split a multi-label cell and strip whitespace."""
    if pd.isna(value):
        return []
    return [t.strip() for t in re.split(seps, str(value)) if t.strip()]


def load_and_clean(path, year_min=2022, year_max=2024):

    df = pd.read_excel(path, sheet_name="Papers")
    df = df[df["Title"].notna()].copy()  # drops trailing totals/empty rows
    # clamp years if needed
    if year_min is not None or year_max is not None:
        yr = pd.to_numeric(df["Year"], errors="coerce")
        lo = year_min if year_min is not None else -float("inf")
        hi = year_max if year_max is not None else float("inf")
        df = df[(yr >= lo) & (yr <= hi)].copy()

    for col in SUITES + ["Microbenchmark", "Custom", "Irrelevant"]:
        # replace nulls with 0s
        df[col] = df[col].fillna(0).astype(int)
        # everything must be 1 or 0 (previously empty)
        bad = df[~df[col].isin([0, 1])]
        if len(bad):
            raise ValueError(f"Non-binary values in '{col}' rows {bad.index.tolist()}")

    # split cells in the 'Other (list)' column into canonical names, using OTHER_ALIASES.
    def _other_tags(v):
        out = set()
        for t in split_tags(v, seps=r"[,;]"):
            canon = OTHER_ALIASES.get(t.lower(), None)
            if canon is not None:
                out.add(canon)
            else:
                raise ValueError(
                    f"Unrecognized other workload '{t}' in cell value '{v}'"
                )
        return sorted(out)

    df["Other_names"] = df["Other (list)"].apply(_other_tags)

    # Workloads: split on , and ; only (keep 'I/O'); drop tags mapped to None.
    def _workload_tags(v):
        out = set()
        for t in split_tags(v, seps=r"[,;]"):
            canon = WORKLOAD_ALIASES.get(t.lower(), None)
            if canon is not None:
                out.add(canon)
            else:
                raise ValueError(f"Unrecognized workload tag '{t}' in cell value '{v}'")
        return sorted(out)

    df["Workload_tags"] = df["Workload"].apply(_workload_tags)
    rel = df[df["Irrelevant"] == 0].copy()
    rel["any_annotation"] = (
        rel[SUITES + ["Microbenchmark", "Custom"]].sum(axis=1)
        + rel["Other_names"].str.len()
    ) > 0
    return rel
