from itertools import chain, tee, product
import random
import math

from aalpy.base.Oracle import Oracle
from aalpy.base.SUL import SUL


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


class WpMethodTSDiffEqOracle(Oracle):
    """
    Implements the Wp-method equivalence oracle, but with a twist. In each equivalence query,
    it retains the previous test suite, calculates the current one and first executes the
    difference between the two test suites, TS_new - TS_old.
    """

    def __init__(self, alphabet: list, sul: SUL, max_number_of_states):
        super().__init__(alphabet, sul)
        self.m = max_number_of_states
        self.cache = set()
        self.prev_hypothesis = None

    def find_cex(self, hypothesis):
        if not hypothesis.characterization_set:
            hypothesis.characterization_set = frozenset(
                hypothesis.compute_characterization_set()
            )

        depth = self.m + 1 - len(hypothesis.states)

        if self.prev_hypothesis:
            # if there is a previous hypothesis, execute the first phase test
            # suite by using the difference of the two state coverage sets as
            # the prefix, instead of the whole new one.
            state_cover = [state.prefix for state in hypothesis.states]

            new = {s.state_id: s for s in hypothesis.states}
            old = {s.state_id: s for s in self.prev_hypothesis.states}
            new_states = [new[s] for s in new if not s in old]
            diff_sc = [s.prefix for s in new_states]

            # remove the new states and prepend them
            state_cover = [s for s in state_cover if s not in diff_sc]
            state_cover = diff_sc + state_cover
            transition_cover = frozenset(
                prefix + (letter,) for prefix in state_cover for letter in self.alphabet
            )
            difference = transition_cover.difference(state_cover)
        else:
            state_cover = frozenset(state.prefix for state in hypothesis.states)
            transition_cover = frozenset(
                state.prefix + (letter,)
                for state in hypothesis.states
                for letter in self.alphabet
            )
            difference = transition_cover.difference(state_cover)

        self.prev_hypothesis = hypothesis
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


class RandomWpMethodEqOracle(Oracle):
    """
    Implements the Random Wp-Method.
    """

    def __init__(
        self, alphabet: list, sul: SUL, expected_length=10, min_length=1, bound=1000
    ):
        super().__init__(alphabet, sul)
        self.expected_length = expected_length
        self.min_length = min_length
        self.bound = bound

    def find_cex(self, hypothesis):
        if not hypothesis.characterization_set:
            hypothesis.characterization_set = hypothesis.compute_characterization_set()

        state_mapping = {
            s: state_characterization_set(hypothesis, self.alphabet, s)
            for s in hypothesis.states
        }

        for _ in range(self.bound):
            state = random.choice(hypothesis.states)
            input = state.prefix
            limit = self.min_length
            while limit > 0 or random.random() > 1 / (self.expected_length + 1):
                letter = random.choice(self.alphabet)
                input += (letter,)
                limit -= 1
            if random.random() > 0.5:
                # global suffix with characterization_set
                input += random.choice(hypothesis.characterization_set)
            else:
                # local suffix
                _ = hypothesis.execute_sequence(hypothesis.initial_state, input)
                input += random.choice(state_mapping[hypothesis.current_state])

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
                state = random.choice(hypothesis.states)
            else:
                group = random.choices(self.age_groups, weights)[0]
                id = random.choice(group)
                state = hypothesis.get_state_by_id(id)
            input = state.prefix
            limit = self.min_length
            while limit > 0 or random.random() > 1 / (self.expected_length + 1):
                letter = random.choice(self.alphabet)
                input += (letter,)
                limit -= 1
            if random.random() > 0.5:
                # global suffix with characterization_set
                input += random.choice(hypothesis.characterization_set)
            else:
                # local suffix
                _ = hypothesis.execute_sequence(hypothesis.initial_state, input)
                input += random.choice(state_mapping[hypothesis.current_state])

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
