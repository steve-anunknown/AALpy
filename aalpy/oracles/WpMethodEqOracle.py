from itertools import chain, tee, product
import random
import math

from aalpy.base.Oracle import Oracle
from aalpy.base.SUL import SUL


def flatten(inlist):
    return [item for sublist in inlist for item in sublist]


def state_characterization_set(hypothesis, alphabet, state):
    """
    Return a list of sequences that distinguish the given state from all other states in the hypothesis.
    Args:
        hypothesis: hypothesis automaton
        alphabet: input alphabet
        state: state for which to find distinguishing sequences
    """
    result = []
    for i in range(len(hypothesis.states)):
        if hypothesis.states[i] == state:
            continue
        seq = hypothesis.find_distinguishing_seq(state, hypothesis.states[i], alphabet)
        if seq:
            result.append(tuple(seq))
    return result


def first_phase_it(alphabet, state_cover, depth, char_set):
    """
    Return an iterator that generates all possible sequences for the first phase of the Wp-method.
    Args:
        alphabet: input alphabet
        state_cover: list of states to cover
        depth: maximum length of middle part
        char_set: characterization set
    """
    char_set = char_set or [()]
    for d in range(depth):
        middle = product(alphabet, repeat=d)
        for m in middle:
            for s in state_cover:
                for c in char_set:
                    yield s + m + c


def second_phase_it(hyp, alphabet, difference, depth):
    """
    Return an iterator that generates all possible sequences for the second phase of the Wp-method.
    Args:
        hyp: hypothesis automaton
        alphabet: input alphabet
        difference: set of sequences that are in the transition cover but not in the state cover
        depth: maximum length of middle part
    """
    state_mapping = {}
    for d in range(depth):
        middle = product(alphabet, repeat=d)
        for mid in middle:
            for t in difference:
                _ = hyp.execute_sequence(hyp.initial_state, t + mid)
                state = hyp.current_state
                if state not in state_mapping:
                    state_mapping[state] = state_characterization_set(
                        hyp, alphabet, state
                    )

                for sm in state_mapping[state]:
                    yield t + mid + sm


class WpMethodEqOracle(Oracle):
    """
    Implements the Wp-method equivalence oracle.
    """

    def __init__(self, alphabet: list, sul: SUL, max_number_of_states=4):
        super().__init__(alphabet, sul)
        self.m = max_number_of_states
        self.cache = set()

    def find_cex(self, hypothesis):
        if not hypothesis.characterization_set:
            hypothesis.characterization_set = hypothesis.compute_characterization_set()

        transition_cover = frozenset(
            state.prefix + (letter,)
            for state in hypothesis.states
            for letter in self.alphabet
        )

        state_cover = frozenset(state.prefix for state in hypothesis.states)
        difference = transition_cover.difference(state_cover)
        depth = self.m + 1 - len(hypothesis.states)
        # first phase State Cover * Middle * Characterization Set
        first_phase = first_phase_it(
            self.alphabet, state_cover, depth, hypothesis.characterization_set
        )

        # second phase (Transition Cover - State Cover) * Middle * Characterization Set
        # of the state that the prefix leads to
        second_phase = second_phase_it(hypothesis, self.alphabet, difference, depth)
        test_suite = chain(first_phase, second_phase)

        for seq in test_suite:
            if not seq in self.cache:
                cex = self.execute_test_case(hypothesis, seq)
                if cex:
                    return cex
                self.cache.add(seq)

        return None


class WpMethodTSDiffEqOracle(Oracle):
    """
    Implements the Wp-method equivalence oracle, but with a twist. In each equivalence query,
    it retains the previous test suite, calculates the current one and first executes the
    difference between the two test suites, TS_new - TS_old.
    """

    def __init__(self, alphabet: list, sul: SUL, max_number_of_states, diff_depth=1):
        super().__init__(alphabet, sul)
        self.m = max_number_of_states
        self.d = diff_depth
        self.age_groups = []
        self.cache = set()

    def find_cex(self, hypothesis):
        if not hypothesis.characterization_set:
            hypothesis.characterization_set = list(
                hypothesis.compute_characterization_set()
            )

        if not self.age_groups:
            self.age_groups.extend([[s.state_id for s in hypothesis.states]])
        else:
            new = []
            for s in hypothesis.states:
                if not any(s.state_id in p for p in self.age_groups):
                    new.append(s.state_id)
            self.age_groups.extend([new])

        depth = self.m + 1 - len(hypothesis.states)

        new_states = [
            hypothesis.get_state_by_id(s)
            for s in flatten(self.age_groups[-1::-1][: self.d])
        ]

        state_cover = [
            state.prefix
            for state in new_states
            + [s for s in hypothesis.states if s not in new_states]
        ]
        transition_cover = [
            prefix + (letter,) for prefix in state_cover for letter in self.alphabet
        ]
        difference = set(transition_cover).difference(state_cover)

        first_phase = first_phase_it(
            self.alphabet, state_cover, depth, hypothesis.characterization_set
        )

        second_phase = second_phase_it(hypothesis, self.alphabet, difference, depth)
        test_suite = chain(first_phase, second_phase)
        for seq in test_suite:
            if seq not in self.cache:
                cex = self.execute_test_case(hypothesis, seq)
                if cex:
                    return cex
                self.cache.add(seq)


class RandomWpMethodEqOracle(Oracle):
    """
    Implements the Random Wp-Method.
    """

    def __init__(
        self,
        alphabet: list,
        sul: SUL,
        expected_length=10,
        min_length=1,
        bound=1000,
        seed=None,
    ):
        super().__init__(alphabet, sul)
        self.expected_length = expected_length
        self.min_length = min_length
        self.bound = bound
        if seed:
            self.rng = self.rng.Random(seed)
        else:
            self.rng = self.rng.Random()

    def find_cex(self, hypothesis):
        if not hypothesis.characterization_set:
            hypothesis.characterization_set = hypothesis.compute_characterization_set()

        state_mapping = {
            s: state_characterization_set(hypothesis, self.alphabet, s)
            for s in hypothesis.states
        }

        for _ in range(self.bound):
            state = self.rng.choice(hypothesis.states)
            input = state.prefix
            limit = self.min_length
            while limit > 0 or self.rng.random() > 1 / (self.expected_length + 1):
                letter = self.rng.choice(self.alphabet)
                input += (letter,)
                limit -= 1
            if self.rng.random() > 0.5:
                # global suffix with characterization_set
                input += self.rng.choice(hypothesis.characterization_set)
            else:
                # local suffix
                _ = hypothesis.execute_sequence(hypothesis.initial_state, input)
                input += self.rng.choice(state_mapping[hypothesis.current_state])

            # execute the sequence
            cex = self.execute_test_case(hypothesis, input)
            if cex:
                return cex
        return None


class StochasticWpMethodEqOracle(Oracle):
    """
    Implements the Random Wp-Method but with a bias towards sampling new
    states.
    """

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
        expected_length=10,
        min_length=1,
        bound=1000,
        prob_function="random",
        seed=None,
    ):
        super().__init__(alphabet, sul)
        self.expected_length = expected_length
        self.min_length = min_length
        self.bound = bound
        self.age_groups = []
        assert prob_function in [
            "linear",
            "square",
            "exponential",
            "random",
        ], "Probability function must be one of 'linear', 'square', 'exponential' or 'random'."
        self.prob_function = (
            getattr(self, prob_function) if prob_function != "random" else "random"
        )
        if seed:
            self.rng = random.Random(seed)
        else:
            self.rng = random.Random()

    def find_cex(self, hypothesis):
        if not hypothesis.characterization_set:
            hypothesis.characterization_set = hypothesis.compute_characterization_set()

        if not self.age_groups:
            self.age_groups.append([s.state_id for s in hypothesis.states])
        else:
            new = []
            for state in hypothesis.states:
                if not any(state.state_id in p for p in self.age_groups):
                    new.append(state.state_id)
            self.age_groups.append(new)

        if not self.prob_function == "random":
            n = len(self.age_groups)
            weights = [self.prob_function(i, n) for i in range(n)]
            total = sum(weights)
            assert math.isclose(
                total, 1
            ), f"Invalid probability function. Probabilities do not sum up to 1 but to {total}."

        state_mapping = {
            s: state_characterization_set(hypothesis, self.alphabet, s)
            for s in hypothesis.states
        }
        for _ in range(self.bound):
            if self.prob_function == "random":
                state = self.rng.choice(hypothesis.states)
            else:
                group = self.rng.choices(self.age_groups, weights)[0]
                id = self.rng.choice(group)
                state = hypothesis.get_state_by_id(id)
            input = state.prefix
            limit = self.min_length
            while limit > 0 or self.rng.random() > 1 / (self.expected_length + 1):
                letter = self.rng.choice(self.alphabet)
                input += (letter,)
                limit -= 1
            if self.rng.random() > 0.5:
                # global suffix with characterization_set
                input += self.rng.choice(hypothesis.characterization_set)
            else:
                # local suffix
                _ = hypothesis.execute_sequence(hypothesis.initial_state, input)
                input += self.rng.choice(state_mapping[hypothesis.current_state])

            # execute the sequence
            cex = self.execute_test_case(hypothesis, input)
            if cex:
                return cex


class WpMethodDiffFirstEqOracle(Oracle):
    """
    Implements the Wp-method equivalence oracle.
    """

    def __init__(self, alphabet: list, sul: SUL, max_number_of_states):
        super().__init__(alphabet, sul)
        self.m = max_number_of_states
        self.cache = set()

    def find_cex(self, hypothesis):
        if not hypothesis.characterization_set:
            hypothesis.characterization_set = hypothesis.compute_characterization_set()

        # keep them as lists to preserver order.
        transition_cover = [
            state.prefix + (letter,)
            for state in reversed(hypothesis.states)
            for letter in self.alphabet
        ]
        state_cover = [state.prefix for state in reversed(hypothesis.states)]
        difference = [el for el in transition_cover if el not in set(state_cover)]
        depth = self.m + 1 - len(hypothesis.states)
        # first phase State Cover * Middle * Characterization Set
        first_phase = first_phase_it(
            self.alphabet, state_cover, depth, hypothesis.characterization_set
        )
        # second phase (Transition Cover - State Cover) * Middle * Characterization Set
        # of the state that the prefix leads to
        second_phase = second_phase_it(hypothesis, self.alphabet, difference, depth)
        test_suite = chain(first_phase, second_phase)
        for seq in test_suite:
            if seq not in self.cache:
                self.reset_hyp_and_sul(hypothesis)
                outputs = []

                for ind, letter in enumerate(seq):
                    out_hyp = hypothesis.step(letter)
                    out_sul = self.sul.step(letter)
                    self.num_steps += 1

                    outputs.append(out_sul)
                    if out_hyp != out_sul:
                        self.sul.post()
                        return seq[: ind + 1]
                self.cache.add(seq)

        return None
