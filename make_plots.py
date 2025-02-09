import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import argparse
import pathlib
import itertools
import os

from aalpy.utils.FileHandler import load_automaton_from_file

PROTOCOLS = ["tls", "mqtt", "tcp", "dtls"]

def keep_successes(queries, failures):
    filtered = []
    for m in range(failures.shape[0]):
        valid = np.all(failures[m] == 0, axis=1)
        filtered.append(queries[m, valid, :])
    return filtered

def compute_scores(queries, failures):
    successful = np.copy(queries)
    # replace failed experiments by nan
    successful[failures == 1] = np.nan
    mask = np.all(np.isnan(successful), axis=1)
    keep = ~np.any(mask, axis=1)
    successful = successful[keep]
    # compute scores ignoring fails
    averages = np.nanmean(successful, axis=1)
    s1_scores = np.sum(averages, axis=0)

    maxima = np.nanmax(averages, axis=1)
    s2_scores = np.sum(averages / maxima[:, np.newaxis], axis=0)
    # compute penalized s2 score by factoring in fails
    fails = np.sum(np.mean(failures,axis=1), axis=0)
    s2_scores_penalized = s2_scores + fails
    return (s1_scores, s2_scores, s2_scores_penalized)


def draw_plots(data, results_dir):
    for score in data.columns:
        for zoom in [False, True]:
            plt.figure(figsize=(8, 6))
            sns.barplot(x=data.index, y=data[score], palette="viridis", hue=data.index)
            plt.title(f'{score} Scores{"" if not zoom else " (zoomed)"}', fontsize=16)
            plt.xlabel("Oracle", fontsize=12)
            plt.ylabel("Score", fontsize=12)
            if zoom:
                if not data[score].max() == data[score].min():
                    maxdiff = data[score].max() - data[score].min()
                    plt.ylim(
                        data[score].min() - 0.1 * maxdiff,
                        data[score].max() + 0.1 * maxdiff,
                    )
            plt.tight_layout()
            name = f'{score.lower()}_scores{"" if not zoom else "_zoomed"}'
            plt.savefig(f"{results_dir}/{name}.pdf", format="pdf")
            plt.close()
    oracles = data.index
    s2 = data["S2"]
    s2p = data["S2_Penalized"]
    new = pd.DataFrame({"oracles": oracles, "S2": s2, "S2_Penalized": s2p})
    melted = new.melt(id_vars="oracles", value_vars=["S2", "S2_Penalized"])
    plt.figure(figsize=(8, 6))
    sns.barplot(x="oracles", y="value", hue="variable", data=melted, palette="viridis")
    plt.title("S2 Scores", fontsize=16)
    plt.xlabel("Oracle", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    maxdiff = max(melted["value"].max() - melted["value"].min(), 0.1)
    plt.ylim(melted["value"].min() - 0.1 * maxdiff, melted["value"].max() + 0.1 * maxdiff)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/s2_scores_together.pdf", format="pdf")
    plt.close()


def make_plots(base_method, results_dir, protocols):
    if base_method == "state_coverage":
        oracles = ["Random", "Linear", "Quadratic", "Exponential"]
    elif base_method == "wmethod":
        oracles = ["Normal", "Reverse"]
    elif base_method == "wpmethod":
        oracles = ["Normal", "Reverse", "TSDiff"] # add more later
    elif base_method == "rwpmethod":
        oracles = ["Normal", "Linear", "Quadratic", "Exponential"]
    else:
        oracles = [
            ["Random", "Linear", "Quadratic", "Exponential", "Inverse"], # state_coverage
            ["Normal", "Reverse"], # wmethod
            ["Normal", "Reverse", "TSDiff"], # wpmethod
            ["Normal", "New First"], # rwpmethod
        ]
    protocols = PROTOCOLS if protocols == "all" else [protocols]
    oracles = oracles if base_method == "all" else [oracles]
    methods = ["state_coverage", "wmethod", "wpmethod", "rwpmethod"] if base_method == "all" else [base_method]
    for method, orcs in zip(methods, oracles):
        if protocols == ["combined"]:
            measurements = np.load(f"{results_dir}/{method}/eq_queries.npy")
            failures = np.load(f"{results_dir}/{method}/failures.npy")
            (s1_scores, s2_scores, s2_scores_penalized) = compute_scores(measurements, failures)
            scores = np.array([s1_scores, s2_scores, s2_scores_penalized]).T
            df = pd.DataFrame(scores, columns=["S1", "S2", "S2_Penalized"], index=orcs)
            draw_plots(df, f"{results_dir}/{method}")
            continue

        for protocol in protocols:
            protocol = protocol.upper()
            curdir = f"{results_dir}/{method}/{protocol}"
            # shape of measurements is (num_models, num_runs, num_oracles)
            measurements = np.load(f"{curdir}/eq_queries.npy")
            failures = np.load(f"{curdir}/failures.npy")
            (s1_scores, s2_scores, s2_scores_penalized) = compute_scores(measurements, failures)
            scores = np.array([s1_scores, s2_scores, s2_scores_penalized]).T
            df = pd.DataFrame(scores, columns=["S1", "S2", "S2_Penalized"], index=orcs)
            draw_plots(df, curdir)
            if protocol == "DTLS":
                # more size-specific results
                modeldir = pathlib.Path("./DotModels/DTLS")
                files = list(modeldir.iterdir())
                models = [load_automaton_from_file(f, 'mealy') for f in files]
                sizes = [m.size for m in models]
                # order which experiments were run in
                indexed = sorted(list(enumerate(sizes)), key=lambda x: x[1])
                groups = itertools.groupby(indexed, key=lambda x: x[1] // 20)
                for k, group in groups:
                    id = (k * 20, (k+1) * 20)
                    indices = np.array(list(map(lambda x: x[0], group)))
                    ms = measurements[indices]
                    fs = failures[indices]
                    (s1_scores, s2_scores, s2_scores_penalized) = compute_scores(ms, fs)
                    scores = np.array([s1_scores, s2_scores, s2_scores_penalized]).T
                    df = pd.DataFrame(scores, columns=["S1", "S2", "S2_Penalized"], index=orcs)
                    if not os.path.exists(f'{curdir}/sizes_{id[0]}_{id[1]}'):
                        os.makedirs(f'{curdir}/sizes_{id[0]}_{id[1]}')
                    draw_plots(df, f'{curdir}/sizes_{id[0]}_{id[1]}')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plotting script")
    parser.add_argument(
        "-r",
        "--results_dir",
        type=str,
        default="results",
        help="Root directory of the results",
        required=True,
    )
    parser.add_argument(
        "-b",
        "--base_method",
        type=str,
        choices=["state_coverage", "wmethod", "wpmethod", "rwpmethod", "all"],
        default="state_coverage",
        help="Results of base method to plot",
        required=True,
    )
    parser.add_argument(
        "-p",
        "--protocols",
        type=str,
        choices=PROTOCOLS + ["combined", "all"],
        default="all",
        help="Protocols to plot",
        required=True,
    )

    args = parser.parse_args()
    base_method = args.base_method
    results_dir = args.results_dir
    protocols = args.protocols

    sns.set_theme()
    sns.set_context("paper")

    make_plots(base_method, results_dir, protocols)
