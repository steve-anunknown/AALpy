import os
import gc
import shutil
import pathlib
import numpy as np
import multiprocessing as mp
from rich.progress import Progress

# import argument parser
import argparse

from aalpy.oracles.WMethodEqOracle import (
    WMethodEqOracle,
    WMethodDiffFirstEqOracle,
    WMethodTSDiffEqOracle,
)
from aalpy.oracles.WpMethodEqOracle import (
    WpMethodEqOracle,
    WpMethodDiffFirstEqOracle,
    WpMethodTSDiffEqOracle,
    StochasticWpMethodEqOracle,
)
from aalpy.oracles.StochasticStateCoverageEqOracle import (
    StochasticStateCoverageEqOracle,
)
from aalpy.SULs.AutomataSUL import AutomatonSUL
from aalpy.learning_algs.deterministic.LStar import run_Lstar
from aalpy.utils.FileHandler import load_automaton_from_file, save_automaton_to_file
from aalpy.utils import bisimilar

# with uniform random
# TCP_Linux_Client is 10240 -> 12000 for good measure
# TCP_Linux_Server is 524288 -> 550000 for good measure
# tcp_server_ubuntu_trans 393216 -> 410000 for good measure
# tcp_server_bsd_trans is 2097152 -> 2200000 for good measure
# tcp_server_windows_trans is 49152 -> 60000 for good measure
TCP_MODELS = {
    "TCP_Linux_Client": 20000,
    "TCP_Linux_Server": 550000,
    "tcp_server_ubuntu_trans": 410000,
    "tcp_server_bsd_trans": 2200000,
    "tcp_server_windows_trans": 60000,
}


# with uniform random
def DTLS_MODELS(size):
    """
    Generate the number of walks per round for the DTLS models.

    Args:
        size: size of the model
    """
    # dtls for 80 < states <= 120 is 524288 -> 550000 for good measure
    if 80 < size <= 120:
        return 550000
    # dtls for 50 < states <= 80 is 262144 -> 300000 for good measure
    elif 50 < size <= 80:
        return 300000
    # dtls for 30 < states <= 50 is 57344 -> 70000 for good measure
    elif 30 < size <= 50:
        return 70000
    # dtls for 20 < states <= 30 is 163840 -> 200000 for good measure
    elif 20 < size <= 30:
        return 200000
    # dtls for 10 <= states <= 20 is 163840 -> 200000 for good measure
    elif 10 <= size <= 20:
        return 200000
    else:
        raise ValueError("Invalid size for DTLS model")


WALKS_PER_ROUND = {
    "TLS": 1,
    "MQTT": 5000,
}

# for exploration = 4
# TLS longest suffix = 1 => therefore random walk length = 5
# TCP longest suffix = 6 => therefore random walk length = 10
# MQTT longest suffix = 4 => therefore random walk length = 8
# DTLS longest suffix = 6 => therefore random walk length = 10
WALK_LEN = {"TCP": 10, "TLS": 5, "MQTT": 8, "DTLS": 10}

METHOD_TO_ORACLES = {
    "wmethod": 6,
    "wpmethod": 6,
    "rwpmethod": 4,
    "state_coverage": 4,
}


def process_oracle(alphabet, sul, oracle, correct_size, i):
    """
    Process the oracle and return the number of queries to the equivalence and membership oracles
    and whether the learned model has the correct size.

    Args:
        alphabet: input alphabet
        sul: system under learning
        oracle: equivalence oracle
        correct_size: correct size of the model
        i: index of the oracle
    """
    model, info = run_Lstar(
        alphabet,
        sul,
        oracle,
        "mealy",
        cache_and_non_det_check=False,
        return_data=True,
        print_level=0,
    )
    return (
        i,
        info["queries_eq_oracle"],
        info["queries_learning"],
        not bisimilar(model, sul.automaton),
        info["intermediate_hypotheses"],
        info["counterexamples"],
        info["queries_per_round"],
    )


def do_learning_experiments(model, prot, trial):
    """
    Perform the learning experiments for the given model and alphabet.

    Args:
        model: model to learn
        alphabet: input alphabet
        prot: protocol name and model name
    """
    alphabet = list(model.get_input_alphabet())
    # create a copy of the SUL for each oracle
    suls = [AutomatonSUL(model) for _ in range(NUM_ORACLES)]
    # initialize the oracles
    if BASE_METHOD == "state_coverage" or BASE_METHOD == "rwpmethod":
        # wl = WALK_LEN[prot[0]]
        if prot[0] == "DTLS":
            wpr = DTLS_MODELS(model.size)
        elif prot[0] == "TCP":
            wpr = TCP_MODELS[prot[1]]
        else:
            wpr = WALKS_PER_ROUND[prot[0]]
        if BASE_METHOD == "state_coverage":
            wl = WALK_LEN[prot[0]]
            eq_oracles = [
                StochasticRandom(
                    alphabet, suls[0], wpr, wl, seed=hash((BASE_METHOD, 0, prot, trial))
                ),
                StochasticLinear(
                    alphabet, suls[1], wpr, wl, seed=hash((BASE_METHOD, 1, prot, trial))
                ),
                StochasticSquare(
                    alphabet, suls[2], wpr, wl, seed=hash((BASE_METHOD, 2, prot, trial))
                ),
                StochasticExponential(
                    alphabet, suls[3], wpr, wl, seed=hash((BASE_METHOD, 3, prot, trial))
                ),
                # StochasticInverse(alphabet, suls[4], wpr, wl),
            ]
        else:
            eq_oracles = [
                RandomWp(
                    alphabet,
                    suls[0],
                    wpr,
                    seed=hash((BASE_METHOD, 0, prot, trial)),
                ),
                LinearWp(
                    alphabet,
                    suls[1],
                    wpr,
                    seed=hash((BASE_METHOD, 1, prot, trial)),
                ),
                SquareWp(
                    alphabet,
                    suls[2],
                    wpr,
                    seed=hash((BASE_METHOD, 2, prot, trial)),
                ),
                ExponentialWp(
                    alphabet,
                    suls[3],
                    wpr,
                    seed=hash((BASE_METHOD, 3, prot, trial)),
                ),
            ]
    elif BASE_METHOD == "wmethod":
        max_size = model.size + 2
        eq_oracles = [
            WMethod(alphabet, suls[0], max_size),
            WMethodTSDiff1(alphabet, suls[1], max_size),
            WMethodTSDiff2(alphabet, suls[2], max_size),
            WMethodTSDiff3(alphabet, suls[3], max_size),
            WMethodTSDiff6(alphabet, suls[4], max_size),
            WMethodDiffFirst(alphabet, suls[5], max_size),
        ]
    elif BASE_METHOD == "wpmethod":
        max_size = model.size + 2
        eq_oracles = [
            WpMethod(alphabet, suls[0], max_size),
            WpMethodTSDiff1(alphabet, suls[1], max_size),
            WpMethodTSDiff2(alphabet, suls[2], max_size),
            WpMethodTSDiff3(alphabet, suls[3], max_size),
            WpMethodTSDiff6(alphabet, suls[4], max_size),
            WpMethodDiffFirst(alphabet, suls[5], max_size),
        ]
    else:
        raise ValueError("Unknown base method")

    assert len(suls) == len(eq_oracles), "Number of oracles and SULs must be the same."
    assert NUM_ORACLES == len(
        eq_oracles
    ), "Number of oracles must be the same as the number of methods."

    if PARALLEL:
        # create the arguments for eache oracle's task
        tasks = [
            (alphabet, sul, oracle, model.size, i)
            for i, (sul, oracle) in enumerate(zip(suls, eq_oracles))
        ]
        workers = min(mp.cpu_count(), len(tasks))
        with mp.Pool(workers) as pool:
            results = pool.starmap(process_oracle, tasks)
        gc.collect()
    else:
        results = [
            process_oracle(alphabet, sul, oracle, model.size, i)
            for i, (sul, oracle) in enumerate(zip(suls, eq_oracles))
        ]

    return results


def clean_results(method):
    """
    Clean the results directory for the given method.

    Args:
        method: method to clean the results for
    """
    for file in pathlib.Path(f"./results/{method}").iterdir():
        if file.is_dir():
            shutil.rmtree(file)
        else:
            file.unlink()


def main():
    ROOT = os.getcwd() + "/DotModels"
    if FAMILY == "ALL":
        PROTOCOLS = ["TLS", "MQTT", "TCP", "DTLS"]
    else:
        PROTOCOLS = [FAMILY]

    clean_results(BASE_METHOD)

    DIRS = [pathlib.Path(ROOT + "/" + prot) for prot in PROTOCOLS]
    FILES = [file for dir in DIRS for file in dir.iterdir()]
    FILES_PER_PROT = {
        prot: len([file for file in DIRS[i].iterdir()])
        for i, prot in enumerate(PROTOCOLS)
    }
    MODELS = (load_automaton_from_file(f, "mealy") for f in FILES)

    EQ_QUERIES = np.zeros((len(FILES), TIMES, NUM_ORACLES))
    MB_QUERIES = np.zeros((len(FILES), TIMES, NUM_ORACLES))
    FAILURES = np.zeros((len(FILES), TIMES, NUM_ORACLES))
    QPR = np.empty((len(FILES), TIMES, NUM_ORACLES), dtype=object)

    MAX_FILENAME = max(list(map(lambda x: len(x.name), FILES)))
    with Progress() as progress:
        bar = progress.add_task("Learning Model ", total=len(FILES) * TIMES)
        # iterate over the models
        for index, (model, file) in enumerate(zip(MODELS, FILES)):
            # these variables can be shared among the processes
            prot = file.parent.stem
            # repeat the experiments to gather statistics
            progress.update(
                bar, description=f"Learning Model {file.name.ljust(MAX_FILENAME)}"
            )
            for trial in range(TIMES):

                results = do_learning_experiments(model, (prot, file.stem), trial)

                for i, eq_queries, mb_queries, failure, hyps, cexs, qpr in results:
                    EQ_QUERIES[index, trial, i] = eq_queries
                    MB_QUERIES[index, trial, i] = mb_queries
                    FAILURES[index, trial, i] = failure
                    QPR[index, trial, i] = qpr

                    if SAVE_INTERMEDIATE_HYPOTHESES:
                        MODEL_RES_DIR = f"./results/{BASE_METHOD}/{prot}/{file.stem}/trial_{trial}/oracle_{i}"
                        if not os.path.exists(MODEL_RES_DIR):
                            os.makedirs(MODEL_RES_DIR)
                        for j, (hyp, cex) in enumerate(zip(hyps, cexs)):
                            save_automaton_to_file(
                                hyp, f"{MODEL_RES_DIR}/h{j}.dot", "dot"
                            )
                            with open(f"{MODEL_RES_DIR}/cex{j}.txt", "w") as f:
                                f.write(str(cex))
                progress.update(bar, advance=1)
        progress.update(bar, completed=len(FILES) * TIMES)

    # store the protocol specific results
    # this works both for FAMILY == "ALL" and for a single family
    prev = 0
    for prot in PROTOCOLS:
        items = FILES_PER_PROT[prot]
        MODEL_RES_DIR = f"./results/{BASE_METHOD}/{prot}"
        if not os.path.exists(MODEL_RES_DIR):
            os.makedirs(MODEL_RES_DIR)
        np.save(
            f"{MODEL_RES_DIR}/eq_queries.npy",
            EQ_QUERIES[prev : prev + items, :, :],
        )
        np.save(
            f"{MODEL_RES_DIR}/mb_queries.npy",
            MB_QUERIES[prev : prev + items, :, :],
        )
        np.save(
            f"{MODEL_RES_DIR}/failures.npy",
            FAILURES[prev : prev + items, :, :],
        )
        np.save(
            f"{MODEL_RES_DIR}/queries_per_round.npy",
            QPR[prev : prev + items, :, :],
        )
        prev += items

    if FAMILY == "ALL":
        # store these results only if all families are learned
        for array, name in zip(
            [EQ_QUERIES, MB_QUERIES, FAILURES], ["eq_queries", "mb_queries", "failures"]
        ):
            averages = np.mean(array, axis=1)
            std_devs = np.std(array, axis=1)

            np.save(f"./results/{BASE_METHOD}/{name}.npy", array)
            np.save(f"./results/{BASE_METHOD}/{name}_averages.npy", averages)
            np.save(f"./results/{BASE_METHOD}/{name}_std_devs.npy", std_devs)
            if not "failures" == name:
                s1_scores = np.sum(averages, axis=0)
                maxima = np.max(averages, axis=1)
                s2_scores = np.sum(averages / maxima[:, np.newaxis], axis=0)

                np.save(f"./results/{BASE_METHOD}/{name}_s1_scores.npy", s1_scores)
                np.save(f"./results/{BASE_METHOD}/{name}_s2_scores.npy", s2_scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse arguments for running learning experiments."
    )

    parser.add_argument(
        "-p",
        "--parallel",
        action="store_true",
        default=False,
        help="Run the experiments in parallel or not. Defaults to False.",
    )

    parser.add_argument(
        "-t",
        "--times",
        type=int,
        default=30,
        help="Number of times to run the stochastic experiments. Defaults to 30.",
    )

    parser.add_argument(
        "-b",
        "--base_method",
        type=str,
        choices=["state_coverage", "wmethod", "wpmethod", "rwpmethod"],
        required=True,
        help="Base method to use.",
    )

    parser.add_argument(
        "-s",
        "--save_intermediate",
        action="store_true",
        default=False,
        help="Save intermediate results or not. Defaults to False.",
    )

    parser.add_argument(
        "-f",
        "--family",
        type=str,
        choices=["tls", "mqtt", "tcp", "dtls", "all"],
        required=True,
        help="Family of models to learn.",
    )

    args = parser.parse_args()
    TIMES = args.times
    PARALLEL = args.parallel
    BASE_METHOD = args.base_method
    SAVE_INTERMEDIATE_HYPOTHESES = args.save_intermediate
    FAMILY = args.family.upper()

    NUM_ORACLES = METHOD_TO_ORACLES[BASE_METHOD]

    if BASE_METHOD == "state_coverage":

        class StochasticRandom(StochasticStateCoverageEqOracle):
            def __init__(self, alphabet, sul, walks_per_round, walk_len, seed=None):
                super().__init__(
                    alphabet,
                    sul,
                    walks_per_round,
                    walk_len,
                    prob_function="random",
                    seed=seed,
                )

        class StochasticLinear(StochasticStateCoverageEqOracle):
            def __init__(self, alphabet, sul, walks_per_round, walk_len, seed=None):
                super().__init__(
                    alphabet,
                    sul,
                    walks_per_round,
                    walk_len,
                    prob_function="linear",
                    seed=seed,
                )

        class StochasticSquare(StochasticStateCoverageEqOracle):
            def __init__(self, alphabet, sul, walks_per_round, walk_len, seed=None):
                super().__init__(
                    alphabet,
                    sul,
                    walks_per_round,
                    walk_len,
                    prob_function="square",
                    seed=seed,
                )

        class StochasticExponential(StochasticStateCoverageEqOracle):
            def __init__(
                self,
                alphabet,
                sul,
                walks_per_round,
                walk_len,
                seed=None,
            ):
                super().__init__(
                    alphabet,
                    sul,
                    walks_per_round,
                    walk_len,
                    prob_function="exponential",
                    seed=seed,
                )

    elif BASE_METHOD == "wmethod":
        TIMES = 1  # WMethod is deterministic

        class WMethod(WMethodEqOracle):
            def __init__(self, alphabet, sul, max_model_size):
                super().__init__(alphabet, sul, max_model_size)

        class WMethodDiffFirst(WMethodDiffFirstEqOracle):
            def __init__(self, alphabet, sul, max_model_size):
                super().__init__(alphabet, sul, max_model_size)

        class WMethodTSDiff1(WMethodTSDiffEqOracle):
            def __init__(self, alphabet, sul, max_model_size, diff_depth=1):
                super().__init__(alphabet, sul, max_model_size, diff_depth)

        class WMethodTSDiff2(WMethodTSDiffEqOracle):
            def __init__(self, alphabet, sul, max_model_size, diff_depth=2):
                super().__init__(alphabet, sul, max_model_size, diff_depth)

        class WMethodTSDiff3(WMethodTSDiffEqOracle):
            def __init__(self, alphabet, sul, max_model_size, diff_depth=3):
                super().__init__(alphabet, sul, max_model_size, diff_depth)

        class WMethodTSDiff6(WMethodTSDiffEqOracle):
            def __init__(self, alphabet, sul, max_model_size, diff_depth=6):
                super().__init__(alphabet, sul, max_model_size, diff_depth)

    elif BASE_METHOD == "wpmethod":
        TIMES = 1

        class WpMethod(WpMethodEqOracle):
            def __init__(self, alphabet, sul, max_model_size):
                super().__init__(alphabet, sul, max_model_size)

        class WpMethodDiffFirst(WpMethodDiffFirstEqOracle):
            def __init__(self, alphabet, sul, max_model_size):
                super().__init__(alphabet, sul, max_model_size)

        class WpMethodTSDiff1(WpMethodTSDiffEqOracle):
            def __init__(self, alphabet, sul, max_model_size, diff_depth=1):
                super().__init__(alphabet, sul, max_model_size, diff_depth)

        class WpMethodTSDiff2(WpMethodTSDiffEqOracle):
            def __init__(self, alphabet, sul, max_model_size, diff_depth=2):
                super().__init__(alphabet, sul, max_model_size, diff_depth)

        class WpMethodTSDiff3(WpMethodTSDiffEqOracle):
            def __init__(self, alphabet, sul, max_model_size, diff_depth=3):
                super().__init__(alphabet, sul, max_model_size, diff_depth)

        class WpMethodTSDiff6(WpMethodTSDiffEqOracle):
            def __init__(self, alphabet, sul, max_model_size, diff_depth=6):
                super().__init__(alphabet, sul, max_model_size, diff_depth)

    elif BASE_METHOD == "rwpmethod":

        class RandomWp(StochasticWpMethodEqOracle):
            def __init__(
                self,
                alphabet,
                sul,
                bound=1000,
                seed=None,
            ):
                super().__init__(
                    alphabet,
                    sul,
                    expected_length=4,
                    min_length=0,
                    bound=bound,
                    prob_function="random",
                    seed=seed,
                )

        class LinearWp(StochasticWpMethodEqOracle):
            def __init__(
                self,
                alphabet,
                sul,
                bound=1000,
                seed=None,
            ):
                super().__init__(
                    alphabet,
                    sul,
                    expected_length=4,
                    min_length=0,
                    bound=bound,
                    prob_function="linear",
                    seed=seed,
                )

        class SquareWp(StochasticWpMethodEqOracle):
            def __init__(
                self,
                alphabet,
                sul,
                bound=1000,
                seed=None,
            ):
                super().__init__(
                    alphabet,
                    sul,
                    expected_length=4,
                    min_length=0,
                    bound=bound,
                    prob_function="square",
                    seed=seed,
                )

        class ExponentialWp(StochasticWpMethodEqOracle):
            def __init__(
                self,
                alphabet,
                sul,
                bound=1000,
                seed=None,
            ):
                super().__init__(
                    alphabet,
                    sul,
                    expected_length=4,
                    min_length=0,
                    bound=bound,
                    prob_function="exponential",
                    seed=seed,
                )

    main()
