import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import argparse

def draw_plots(data, results_dir):
    for score in data.columns:
        for zoom in [False, True]:
            plt.figure(figsize=(8, 6))
            sns.barplot(
                x=data.index, y=data[score], palette="viridis", hue=data.index
            )
            plt.title(
                f'{score} Scores{"" if not zoom else " (zoomed)"}', fontsize=16
            )
            plt.xlabel("Oracle", fontsize=12)
            plt.ylabel("Score", fontsize=12)
            if zoom:
                if not data[score].max() == data[score].min():
                    maxdiff = data[score].max() - data[score].min()
                    plt.ylim(data[score].min() - 0.1 * maxdiff, data[score].max() + 0.1 * maxdiff)
            plt.tight_layout()
            name = f'{score.lower()}_scores{"" if not zoom else "_zoomed"}'
            plt.savefig(f"{results_dir}/{name}.pdf", format="pdf")
            plt.close()

def make_plots(base_method, results_dir, protocols):
    if base_method == "state_coverage":
        oracles = ["Random", "Linear", "Quadratic", "Exponential", "Inverse"]
    elif base_method == "wmethod":
        oracles = ["Normal", "Reverse"]
    elif base_method == "wpmethod":
        oracles = ["Normal", "Reverse"] # add more later
    else:
        oracles = [
            ["Random", "Linear", "Quadratic", "Exponential", "Inverse"], # state_coverage
            ["Normal", "Reverse"], # wmethod
            ["Normal", "Reverse"], # wpmethod
        ]
    protocols = ["tls", "mqtt", "tcp"] if protocols == "all" else [protocols]
    oracles = oracles if base_method == "all" else [oracles]
    methods = ["state_coverage", "wmethod", "wpmethod"] if base_method == "all" else [base_method]
    for method, orcs in zip(methods, oracles):
        if protocols == ["combined"]:
            s1_scores = np.load(f"{results_dir}/{method}/eq_queries_s1_scores.npy")
            s2_scores = np.load(f"{results_dir}/{method}/eq_queries_s2_scores.npy")
            scores = np.array([s1_scores, s2_scores]).T
            df = pd.DataFrame(scores, columns=["S1", "S2"], index=orcs)
            draw_plots(df, f'{results_dir}/{method}')
            continue

        for protocol in protocols:
            protocol = protocol.upper()
            curdir = f"{results_dir}/{method}/{protocol}"
            # shape of measurements is (num_models, num_runs, num_oracles)
            measurements = np.load(f"{curdir}/eq_queries.npy")
            averages = np.mean(measurements, axis=1)
            s1_scores = np.sum(averages, axis=0)

            maxima = np.max(averages, axis=1)
            s2_scores = np.sum(averages / maxima[:, np.newaxis], axis=0)

            scores = np.array([s1_scores, s2_scores]).T
            df = pd.DataFrame(scores, columns=["S1", "S2"], index=orcs)
            draw_plots(df, curdir)


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
        choices=["state_coverage", "wmethod", "wpmethod", "all"],
        default="state_coverage",
        help="Results of base method to plot",
        required=True,
    )
    parser.add_argument(
        "-p",
        "--protocols",
        type=str,
        choices=["tls", "mqtt", "tcp", "combined", "all"],
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
