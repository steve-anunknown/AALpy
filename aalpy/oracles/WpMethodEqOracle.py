from itertools import chain, tee, product
import random

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

        state_mapping = {}
        for state in hypothesis.states:
            char_set = state_characterization_set(hypothesis, self.alphabet, state)
            state_mapping[state] = char_set

        for _ in range(self.bound):
            state = random.choice(hypothesis.states)
            input = state.prefix
            limit = self.min_length
            while limit > 0 or random.random() > 1 / (self.expected_length + 1):
                letter = random.choice(self.alphabet)
                input += (letter,)
                limit -= 1
            if random.random() > 0.5:
                # global suffix
                if not hypothesis.characterization_set:
                    continue
                input += random.choice(hypothesis.characterization_set)
            else:
                # local suffix
                _ = hypothesis.execute_sequence(hypothesis.initial_state, input)
                current_state = hypothesis.current_state
                if not state_mapping[current_state]:
                    continue
                input += random.choice(state_mapping[current_state])

            # execute the sequence
            self.reset_hyp_and_sul(hypothesis)
            outputs = []
            for ind, letter in enumerate(input):
                out_hyp = hypothesis.step(letter)
                out_sul = self.sul.step(letter)
                self.num_steps += 1

                outputs.append(out_sul)
                if out_hyp != out_sul:
                    self.sul.post()
                    return input[: ind + 1]
        return None


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
