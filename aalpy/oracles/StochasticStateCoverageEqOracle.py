import random
import collections
import math

from aalpy.base.Oracle import Oracle
from aalpy.base.SUL import SUL

probability_functions = ["linear", "exponential", "square", "random", "user"]


class StochasticStateCoverageEqOracle(Oracle):
    def linear(self, x, size):
        fundamental = 2 / (size * (size + 1))
        return (x + 1) * fundamental

    def square(self, x, size):
        fundamental = 6 / ((2 * size + 1) * size * (size + 1))
        return ((x + 1) ** 2) * fundamental

    def exponential(self, x, size):
        fundamental = 1 / (2**size - 1)
        return (2**x) * fundamental

    def __init__(
        self,
        alphabet: list,
        sul: SUL,
        walks_per_round,
        walk_len,
        prob_function,
        user=None,
        seed=None,
    ):
        """
        This oracle uses a probability function to sample groups of states, based on their age.


        Args:

            alphabet: input alphabet

            sul: system under learning

            walks_per_round: maximum number of walks in an equivalence query

            walk_len: length of random walk

            prob_function: either 'linear', 'square', 'exponential' or 'random'

            user: user defined probability function

            seed: seed for random number generator

        """
        assert (
            prob_function in probability_functions
        ), f"Probability function must be one of {probability_functions}"
        super().__init__(alphabet, sul)
        if prob_function == "user":
            assert (
                user is not None
            ), "User defined probability function must be provided."
            self.prob_function = user
        elif not prob_function == "random":
            self.prob_function = getattr(self, prob_function)
        else:
            self.prob_function = "random"
        self.steps_per_walk = walk_len
        self.walks_per_round = walks_per_round
        self.age_groups = collections.deque()
        if seed:
            self.rng = random.Random(seed)
        else:
            self.rng = random.Random()

    def find_cex(self, hypothesis):
        if not self.age_groups:
            self.age_groups.extend([[s.state_id for s in hypothesis.states]])
        else:
            new = []
            for s in hypothesis.states:
                if not any(s.state_id in p for p in self.age_groups):
                    new.append(s.state_id)
            self.age_groups.extend([new])
        if not self.prob_function == "random":
            n = len(self.age_groups)
            probabilities = [self.prob_function(i, n) for i in range(n)]
            assert (
                math.isclose(sum(probabilities) , 1)
            ), "Invalid probability function. Probabilities do not sum up to 1."

        for state in hypothesis.states:
            if state.prefix is None:
                state.prefix = hypothesis.get_shortest_path(
                    hypothesis.initial_state, state
                )

        for _ in range(self.walks_per_round):
            self.reset_hyp_and_sul(hypothesis)

            # sample according to the list of probabilities
            if not self.prob_function == "random":
                group = self.rng.choices(self.age_groups, probabilities)[0]
                state = self.rng.choice(group)
                state = hypothesis.get_state_by_id(state)
            else:
                state = self.rng.choice(hypothesis.states)

            prefix = state.prefix
            for p in prefix:
                hypothesis.step(p)
                self.sul.step(p)
                self.num_steps += 1

            suffix = []
            while self.rng.random() > 1 / (self.steps_per_walk + 1):
                suffix.append(self.rng.choice(self.alphabet))

                out_sul = self.sul.step(suffix[-1])
                out_hyp = hypothesis.step(suffix[-1])
                self.num_steps += 1

                if out_sul != out_hyp:
                    self.sul.post()
                    return prefix + tuple(suffix)
        return None

class StochasticStateCoverageOrderedEqOracle(Oracle):
    def linear(self, x, size):
        fundamental = 2 / (size * (size + 1))
        return (x + 1) * fundamental

    def square(self, x, size):
        fundamental = 6 / ((2 * size + 1) * size * (size + 1))
        return ((x + 1) ** 2) * fundamental

    def exponential(self, x, size):
        fundamental = 1 / (2**size - 1)
        return (2**x) * fundamental

    def __init__(
        self,
        alphabet: list,
        sul: SUL,
        walks_per_round,
        walk_len,
        prob_function,
        user=None,
        seed=None,
    ):
        """
        This oracle uses a probability function to sample groups of states, based on their age.


        Args:

            alphabet: input alphabet

            sul: system under learning

            walks_per_round: maximum number of walks in an equivalence query

            walk_len: length of random walk

            prob_function: either 'linear', 'square', 'exponential' or 'random'

            user: user defined probability function

            seed: seed for random number generator

        """
        assert (
            prob_function in probability_functions
        ), f"Probability function must be one of {probability_functions}"
        super().__init__(alphabet, sul)
        if prob_function == "user":
            assert (
                user is not None
            ), "User defined probability function must be provided."
            self.prob_function = user
        elif not prob_function == "random":
            self.prob_function = getattr(self, prob_function)
        else:
            self.prob_function = "random"
        self.steps_per_walk = walk_len
        self.walks_per_round = walks_per_round
        self.age_groups = collections.deque()
        if seed:
            self.rng = random.Random(seed)
        else:
            self.rng = random.Random()

    def find_cex(self, hypothesis):
        if not self.age_groups:
            self.age_groups.extend([[s.state_id for s in hypothesis.states]])
        else:
            new = []
            for s in hypothesis.states:
                if not any(s.state_id in p for p in self.age_groups):
                    new.append(s.state_id)
            self.age_groups.extend([new])
        if not self.prob_function == "random":
            n = len(self.age_groups)
            probabilities = [self.prob_function(i, n) for i in range(n)]
            assert (
                math.isclose(sum(probabilities) , 1)
            ), "Invalid probability function. Probabilities do not sum up to 1."

        for state in hypothesis.states:
            if state.prefix is None:
                state.prefix = hypothesis.get_shortest_path(
                    hypothesis.initial_state, state
                )

        queries_per_group = [p * self.walks_per_round for p in probabilities]
        assert (
                math.isclose(sum(queries_per_group), self.walks_per_round)
            ), "Invalid probability function. Queries per group do not sum up to total number of queries."
        for queries, group in reversed(list(zip(queries_per_group, self.age_groups))):
            for _ in range(int(queries)):
                self.reset_hyp_and_sul(hypothesis)

                state_id = self.rng.choice(group)
                state = hypothesis.get_state_by_id(state_id)

                prefix = state.prefix
                for p in prefix:
                    hypothesis.step(p)
                    self.sul.step(p)
                    self.num_steps += 1

                suffix = []
                while self.rng.random() > 1 / (self.steps_per_walk + 1):
                    suffix.append(self.rng.choice(self.alphabet))

                    out_sul = self.sul.step(suffix[-1])
                    out_hyp = hypothesis.step(suffix[-1])
                    self.num_steps += 1

                    if out_sul != out_hyp:
                        self.sul.post()
                        return prefix + tuple(suffix)
        return None
