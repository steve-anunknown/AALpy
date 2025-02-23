import numpy as np
import pandas as pd
from uniplot import plot, histogram


def setup(method, prot):
    assert method in ["state_coverage", "wmethod", "wpmethod", "rwpmethod"]
    assert prot in ["TLS", "MQTT", "TCP", "DTLS"]
    queries = np.load(f"results/{method}/{prot}/eq_queries.npy")
    failures = np.load(f"results/{method}/{prot}/failures.npy")
    correct = np.where(failures == 0, queries, np.nan)
    return (queries, failures, correct)


def quickplot_s1(method, prot):
    _, _, correct = setup(method, prot)
    average = np.nanmean(correct, axis=1)
    s1 = np.sum(average, axis=0)
    plot(s1, title=f"s1 scores for {method} {prot}")


def quickplot_s2(method, prot):
    _, failures, correct = setup(method, prot)
    average = np.nanmean(correct, axis=1)
    maxima = np.nanmax(average, axis=1)
    s2 = np.sum(average / maxima[:, np.newaxis], axis=0)
    s2p = s2 + np.sum(np.mean(failures, axis=1), axis=0)
    plot(
        [s2, s2p],
        title=f"s2 scores for {method} {prot}",
        legend_labels=["s2", "s2 penalized"],
    )


def mistakes(method, prot):
    _, failures, _ = setup(method, prot)
    return np.count_nonzero(failures, axis=1)
