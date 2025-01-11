import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import argparse

def make_plots(base_method, results_dir, protocols):
    if base_method == "state_coverage":
        oracles = ["Random", "Linear", "Quadratic", "Exponential", "Inverse"]
    elif base_method == "wmethod":
        oracles = ["Normal", "Reverse"]
    else:
        oracles = [
            ["Random", "Linear", "Quadratic", "Exponential", "Inverse"],
            ["Normal", "Reverse"],
        ]
    protocols = ["tls", "mqtt", "tcp"] if protocols == "all" else [protocols]
    oracles = oracles if base_method == "all" else [oracles]
    methods = ["state_coverage", "wmethod"] if base_method == "all" else [base_method]
    for method, orcs in zip(methods, oracles):
        for protocol in protocols:
            protocol = protocol.upper()
            curdir = f"{results_dir}/{method}/{protocol}"
            # shape of measurements is (num_models, num_runs, num_oracles)
            measurements = np.load(f"{curdir}/eq_queries_{protocol}.npy")
            averages = np.mean(measurements, axis=1)
            s1_scores = np.sum(averages, axis=0)

            maxima = np.max(averages, axis=1)
            s2_scores = np.sum(averages / maxima[:, np.newaxis], axis=0)

            scores = np.array([s1_scores, s2_scores]).T
            df = pd.DataFrame(scores, columns=["S1", "S2"], index=orcs)

            for score in ["S1", "S2"]:
                for zoom in [False, True]:
                    plt.figure(figsize=(8, 6))
                    sns.barplot(
                        x=df.index, y=df[score], palette="viridis", hue=df.index
                    )
                    plt.title(
                        f'{score} Scores{"" if not zoom else " (zoomed)"}', fontsize=16
                    )
                    plt.xlabel("Method", fontsize=12)
                    plt.ylabel("Score", fontsize=12)
                    if zoom:
                        maxdiff = df[score].max() - df[score].min()
                        plt.ylim(df[score].min() - 0.1 * maxdiff, df[score].max() + 0.1 * maxdiff)
                    plt.tight_layout()
                    name = f'{score.lower()}_scores{"" if not zoom else "_zoomed"}'
                    plt.savefig(f"{curdir}/{name}.svg", format="svg")
                    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plotting script")
    # Add arguments
    # root directory of the results
    parser.add_argument(
        "-r",
        "--results_dir",
        type=str,
        default="results",
        help="Root directory of the results",
        required=True,
    )
    # base method : either 'state_coverage' 'wmethod' or 'all'
    parser.add_argument(
        "-b",
        "--base_method",
        type=str,
        choices=["state_coverage", "wmethod", "all"],
        default="state_coverage",
        help="Results of base method to plot",
        required=True,
    )
    # protocols to plot : 'tls' 'mqtt' 'tcp' 'all'
    parser.add_argument(
        "-p",
        "--protocols",
        type=str,
        choices=["tls", "mqtt", "tcp", "all"],
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
