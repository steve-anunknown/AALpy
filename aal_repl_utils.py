import numpy as np
import pandas as pd
import pathlib as pl
from uniplot import plot, histogram
from aalpy.utils.FileHandler import load_automaton_from_file

BASE_METHODS = [
        "state_coverage",
        "wmethod",
        "wpmethod",
        "rwpmethod"
        ]

VARIATIONS = {
    "state_coverage": ["Random", "Linear", "Square", "Exponential"],
    "wmethod": ["Normal", "Reversed"],
    "wpmethod": ["Normal", "Reversed", "TSDiff"],
    "rwpmethod": ["Random", "Linear", "Square", "Exponential"],
}

ROOT = pl.Path("DotModels")
PROTOCOLS = ["TLS", "MQTT", "TCP", "DTLS"]
MODEL_FILES = {p: list(ROOT.joinpath(p).glob("*.dot")) for p in PROTOCOLS}
MODEL_NAMES = {p: [f.stem for f in files] for p, files in MODEL_FILES.items()}
MODELS = {
    p: [load_automaton_from_file(f, "mealy") for f in files]
    for p, files in MODEL_FILES.items()
}


QUERIES = {
        m : {
            p : np.load(f"results/{m}/{p}/eq_queries.npy")
            for p in PROTOCOLS
        } for m in BASE_METHODS
}

FAILURES = {
        m : {
            p : np.load(f"results/{m}/{p}/failures.npy")
            for p in PROTOCOLS
        } for m in BASE_METHODS
}

def loop_degree(model):
    inputs = len(model.get_input_alphabet())
    state_ids = [s.state_id for s in model.states]
    degrees = [len(s.get_same_state_transitions()) / inputs for s in model.states]
    df = pd.DataFrame({"State ID": state_ids, "Loop Degree": degrees})
    return df

def flatten(alist):
    return [item for sublist in alist for item in sublist]

# create a dictionary from model names to the loop degree dataframes
LOOP_DEGREES = {name: loop_degree(model) for name, model in zip(flatten(MODEL_NAMES.values()), flatten(MODELS.values()))}


def setup(method, prot):
    assert method in BASE_METHODS
    assert prot in PROTOCOLS
    queries = QUERIES[method][prot]
    failures = FAILURES[method][prot]
    correct = np.where(failures == 0, queries, np.nan)
    return (queries, failures, correct)


def quickplot_s1(method, prot):
    _, _, correct = setup(method, prot)
    average = np.nanmean(correct, axis=1)
    for i in range(average.shape[0]):
        for j in range(average.shape[1]):
            if np.isnan(average[i, j]):
                average[i, j] = np.nanmax(average[i, :])
    s1 = np.sum(average, axis=0)
    plot(s1, title=f"s1 scores for {method} {prot}")

def quickplot_s1_no_last_rounds(method, prot):
    _, failures, correct = setup(method, prot)
    qpr = np.load(f'results/{method}/{prot}/queries_per_round.npy', allow_pickle=True)
    last_rounds = np.vectorize(lambda x: x[-1])(qpr)
    correct_last_rounds = np.where(failures == 0, last_rounds, np.nan)
    average = np.nanmean(correct, axis=1)
    for i in range(average.shape[0]):
        for j in range(average.shape[1]):
            if np.isnan(average[i, j]):
                average[i, j] = np.nanmax(average[i, :])
    s1 = np.sum(average, axis=0)
    correction = np.nanmean(correct_last_rounds, axis=1)
    for i in range(correction.shape[0]):
        for j in range(correction.shape[1]):
            if np.isnan(correction[i, j]):
                correction[i, j] = np.nanmax(correction[i, :])
    correction = np.sum(correction, axis=0)
    s1_prime = s1 - correction
    plot(s1_prime, title=f"s1 scores for {method} {prot} without last rounds")


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


def model_info(prot):
    names = MODEL_NAMES[prot]
    models = MODELS[prot]
    sizes = [m.size for m in models]
    alphabet_sizes = [len(m.get_input_alphabet()) for m in models]
    longest_prefixes = [max([len(s.prefix) for s in m.states]) for m in models]
    longest_suffixes = [max([len(seq) for seq in m.characterization_set]) for m in models]
    df = pd.DataFrame(
            {
                "Model": names,
                "Size": sizes,
                "Alphabet Size": alphabet_sizes,
                "Longest Prefix": longest_prefixes,
                "Longest Suffix": longest_suffixes
            }
    )
    return df


def pretty_mistakes(method, prot):
    mistakes_ = mistakes(method, prot)
    models = model_info(prot)
    df = pd.concat(
        [models, pd.DataFrame(mistakes_, columns=VARIATIONS[method])], axis=1
    )
    return df

def remove_outliers(data):
    methods = data.columns[3:]
    outliers = data[methods].apply(lambda row: np.sum(row > 10), axis=1)
    return data[outliers == 0]

def hardness_score(model):
    k = len(model.get_input_alphabet())
    p_max = max([len(s.prefix) for s in model.states])
    w_max = max([len(seq) for seq in model.characterization_set])
    m = p_max + w_max
    n = model.size
    learn_hardness = (k * (n ** 2) + n * np.log2(m)) * (n + m)
    test_chance = 1 / (p_max * k ** w_max)
    return learn_hardness / test_chance

def test_chance(model):
    k = len(model.get_input_alphabet())
    p_max = max([len(s.prefix) for s in model.states])
    w_max = max(model.characterization_set)
    return 1 / (p_max * k ** w_max)

def learn_hardness(model):
    k = len(model.get_input_alphabet())
    p_max = max([len(s.prefix) for s in model.states])
    w_max = max(list(map(len, model.characterization_set)))
    m = p_max + w_max
    n = model.size
    return (k * n ** 2 + n * np.log2(m)) * (n + m)

for p, models in MODELS.items():
    for model in models:
        model.characterization_set = model.compute_characterization_set()
        model.compute_prefixes()

MODEL_HARDNESS = {
        name: hardness_score(model)
        for name, model in zip(flatten(MODEL_NAMES.values()), flatten(MODELS.values()))
}
MODEL_HARDNESS = pd.DataFrame(MODEL_HARDNESS.items(), columns=["Model", "Hardness"])
