# this is a python script that computes the lowest upper bound of queries
# so that a specific non deterministic equivalence oracle can confidently
# learn a family of models. This is the case when the oracle correctly
# learns the model 30 times in a row.

# the lowest upper bound is computed by starting from a super low bound
# and doubling it while the oracle is not confident enough. Once the oracle
# is confident enough in an upper bound, the algorithm will perform a binary
# search betweem the last upper bound and the current upper bound to find
# the lowest upper bound.

import argparse
import os
import pathlib
from aalpy.utils.FileHandler import load_automaton_from_file
from aalpy.learning_algs.deterministic.LStar import run_Lstar
from aalpy.oracles.StochasticStateCoverageEqOracle import (
    StochasticStateCoverageEqOracle,
)
from aalpy.SULs.AutomataSUL import AutomatonSUL
from rich.progress import Progress

PROTOCOLS = ["tls", "mqtt", "tcp", "dtls"]
WALK_LEN = {"tcp": 50, "tls": 10, "mqtt": 15, "dtls": 40}
ORACLES = ["state_coverage"]
TRIALS = 30
FORBIDDEN = {"tls": 2**10, "mqtt": 2**14, "tcp": 2**22, "dtls": 2**20}
FORBIDDEN_POWERS = {"tls": 10, "mqtt": 14, "tcp": 22, "dtls": 20}
VARIANTS = ["linear", "exponential", "square", "random"]

def run_experiment(models, lower_bound, walk_len, variant=None):
    """Run the experiment for a given lower bound and walk length

    models -- The models to learn
    lower_bound -- The lower bound of queries
    walk_len -- The length of the random walk
    return -- True if the experiment was successful, False otherwise
    """
    for _ in range(TRIALS):
        for _, model in enumerate(models):
            correct_size = model.size
            alphabet = model.get_input_alphabet()
            sul = AutomatonSUL(model)
            eq_oracle = StochasticStateCoverageEqOracle(
                alphabet,
                sul,
                walks_per_round=lower_bound,
                walk_len=walk_len,
                prob_function=variant,
            )
            learned_model = run_Lstar(
                alphabet,
                sul,
                eq_oracle,
                automaton_type="mealy",
                cache_and_non_det_check=False,
                print_level=0,
            )
            failure = learned_model.size != correct_size
            if failure:
                return False
        
    return True


def main(protocol, oracle, variant=None):
    walk_len = WALK_LEN[protocol]
    forbidden = FORBIDDEN[protocol]
    power = FORBIDDEN_POWERS[protocol]
    # the models are in the DotModels directory
    protocol = protocol.upper()
    if not os.path.exists(f"DotModels/{protocol}"):
        raise FileNotFoundError(f"DotModels/{protocol} does not exist")
    directory = pathlib.Path(f"DotModels/{protocol}")
    models = [load_automaton_from_file(f, "mealy") for f in directory.iterdir()]
    with Progress() as progress:
        outer_bar = progress.add_task("Searching for upper bound", total=power)
        lower_bound = 1
        success = False
        for lower_bound in (2 ** i for i in range(power)):
            success = run_experiment(models, lower_bound, walk_len, variant)
            upper_bound = lower_bound
            if success:
                progress.update(outer_bar, completed=power)
                break
            progress.update(outer_bar, advance=1)

    if upper_bound >= forbidden:
        print(f"Upper bound greater than {forbidden}")
        return

    # binary search
    lower_bound = upper_bound // 2
    epsilon = upper_bound // 100
    delta = (upper_bound - lower_bound) // 2
    while lower_bound < upper_bound and delta >= epsilon:
        middle = (lower_bound + upper_bound) // 2
        print(f"Middle: {middle}", end="\r", flush=True)
        success = run_experiment(models, middle, walk_len, variant)
        if success:
            upper_bound = middle
        else:
            lower_bound = middle + 1
        delta = abs(middle - (lower_bound + upper_bound) // 2)

    print(f"Infimum number of queries for {protocol} using {oracle} is {upper_bound}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find the infimum number of queries for a given oracle to learn a family of models"
    )
    # require protocol. it must be one of the protocols in PROTOCOLS
    parser.add_argument(
        "-p",
        "--protocol",
        type=str,
        required=True,
        choices=PROTOCOLS,
        help="The protocol to learn",
    )
    # require oracle. it must be one of the oracles in ORACLES
    parser.add_argument(
        "-o",
        "--oracle",
        type=str,
        required=True,
        choices=ORACLES,
        help="The oracle to use",
    )
    # if oracle is state_coverage, require a variant
    parser.add_argument(
        "-v",
        "--variant",
        type=str,
        required=False,
        choices=VARIANTS,
        help="The variant of the state coverage oracle",
    )
    args = parser.parse_args()
    main(args.protocol, args.oracle, args.variant)
