from itertools import product
from functools import reduce
from random import shuffle, choice, randint

from aalpy.base.Oracle import Oracle
from aalpy.base.SUL import SUL
from itertools import product


def flatten(inlist):
    return [item for sublist in inlist for item in sublist]


class WMethodEqOracle(Oracle):
    """
    Equivalence oracle based on characterization set/ W-set. From 'Tsun S. Chow.   Testing software design modeled by
    finite-state machines'.
    """

    def __init__(
        self, alphabet: list, sul: SUL, max_number_of_states, shuffle_test_set=True
    ):
        """
        Args:

            alphabet: input alphabet
            sul: system under learning
            max_number_of_states: maximum number of states in the automaton
            shuffle_test_set: if True, test cases will be shuffled
        """

        super().__init__(alphabet, sul)
        self.m = max_number_of_states
        self.shuffle = shuffle_test_set
        self.cache = set()

    def test_suite(self, cover, depth, char_set):
        """
        Construct the test suite for the W Method using
        the provided state cover and characterization set,
        exploring up to a given depth.
        Args:

            cover: list of states to cover
            depth: maximum length of middle part
            char_set: characterization set
        """
        # fix the length of the middle part per loop
        # to avoid generating large sequences early on
        char_set = char_set or [()]
        for d in range(depth):
            middle = product(self.alphabet, repeat=d)
            for m in middle:
                for s in cover:
                    for c in char_set:
                        yield s + m + c

    def find_cex(self, hypothesis):

        if not hypothesis.characterization_set:
            hypothesis.characterization_set = hypothesis.compute_characterization_set()

        # covers every transition of the specification at least once.
        transition_cover = [
            state.prefix + (letter,)
            for state in hypothesis.states
            for letter in self.alphabet
        ]

        depth = self.m + 1 - len(hypothesis.states)
        for seq in self.test_suite(
            transition_cover, depth, hypothesis.characterization_set
        ):
            if seq not in self.cache:
                cex = self.execute_test_case(hypothesis, seq)
                if cex:
                    return cex
                self.cache.add(seq)

        return None


class WMethodDiffFirstEqOracle(Oracle):

    def __init__(self, alphabet: list, sul: SUL, max_number_of_states=4):
        """
        Args:

            alphabet: input alphabet
            sul: system under learning
            max_number_of_states: maximum number of states in the automaton
        """

        super().__init__(alphabet, sul)
        self.m = max_number_of_states
        self.cache = set()

    def test_suite(self, cover, depth, char_set):
        """
        Construct the test suite for the W Method using
        the provided state cover and characterization set,
        exploring up to a given depth.
        Args:

            cover: list of states to cover
            depth: maximum length of middle part
            char_set: characterization set
        """
        # fix the length of the middle part per loop
        # to avoid generating large sequences early on
        char_set = char_set or [()]
        for d in range(depth):
            middle = product(self.alphabet, repeat=d)
            for m in middle:
                for s in cover:
                    for c in char_set:
                        yield s + m + c

    def find_cex(self, hypothesis):
        if not hypothesis.characterization_set:
            hypothesis.characterization_set = hypothesis.compute_characterization_set()

        # covers every transition of the specification at least once.
        # with emphasis on newer states, notice the reversed order of states
        transition_cover = [
            state.prefix + (letter,)
            for state in reversed(hypothesis.states)
            for letter in self.alphabet
        ]
        depth = self.m + 1 - len(hypothesis.states)
        for seq in self.test_suite(
            transition_cover, depth, hypothesis.characterization_set
        ):
            if seq not in self.cache:
                cex = self.execute_test_case(hypothesis, seq)
                if cex:
                    return cex
                self.cache.add(seq)

        return None


class WMethodTSDiffEqOracle(Oracle):

    def __init__(self, alphabet: list, sul: SUL, max_number_of_states=4, diff_depth=1):
        """
        Args:

            alphabet: input alphabet
            sul: system under learning
            max_number_of_states: maximum number of states in the automaton
            diff_depth: how many new age groups will be tested first
        """

        super().__init__(alphabet, sul)
        self.m = max_number_of_states
        self.d = diff_depth
        self.cache = set()
        self.age_groups = []

    def test_suite(self, cover, depth, char_set):
        """
        Construct the test suite for the W Method using
        the provided state cover and characterization set,
        exploring up to a given depth.
        Args:

            cover: list of states to cover
            depth: maximum length of middle part
            char_set: characterization set
        """
        # fix the length of the middle part per loop
        # to avoid generating large sequences early on
        char_set = char_set or [()]
        for d in range(depth):
            middle = product(self.alphabet, repeat=d)
            for m in middle:
                for s in cover:
                    for c in char_set:
                        yield s + m + c

    def find_cex(self, hypothesis):
        if not self.age_groups:
            self.age_groups.extend([[s.state_id for s in hypothesis.states]])
        else:
            new = []
            for s in hypothesis.states:
                if not any(s.state_id in p for p in self.age_groups):
                    new.append(s.state_id)
            self.age_groups.extend([new])

        if not hypothesis.characterization_set:
            hypothesis.characterization_set = hypothesis.compute_characterization_set()

        new_states = [
            hypothesis.get_state_by_id(s)
            for s in flatten(self.age_groups[-1::-1][: self.d])
        ]
        transition_cover = [
            state.prefix + (letter,)
            for state in new_states
            + [s for s in hypothesis.states if not s in new_states]
            for letter in self.alphabet
        ]
        depth = self.m + 1 - len(hypothesis.states)
        for seq in self.test_suite(
            transition_cover, depth, hypothesis.characterization_set
        ):
            if seq not in self.cache:
                cex = self.execute_test_case(hypothesis, seq)
                if cex:
                    return cex
                self.cache.add(seq)

        return None


class RandomWMethodEqOracle(Oracle):
    """
    Randomized version of the W-Method equivalence oracle.
    Random walks stem from fixed prefix (path to the state). At the end of the random
    walk an element from the characterization set is added to the test case.
    """

    def __init__(self, alphabet: list, sul: SUL, walks_per_state=12, walk_len=12):
        """
        Args:

            alphabet: input alphabet

            sul: system under learning

            walks_per_state: number of random walks that should start from each state

            walk_len: length of random walk
        """

        super().__init__(alphabet, sul)
        self.walks_per_state = walks_per_state
        self.random_walk_len = walk_len
        self.freq_dict = dict()

    def find_cex(self, hypothesis):

        if not hypothesis.characterization_set:
            hypothesis.characterization_set = hypothesis.compute_characterization_set()
            # fix for non-minimal intermediate hypothesis that can occur in KV
            if not hypothesis.characterization_set:
                hypothesis.characterization_set = [
                    (a,) for a in hypothesis.get_input_alphabet()
                ]

        states_to_cover = []
        for state in hypothesis.states:
            if state.prefix is None:
                state.prefix = hypothesis.get_shortest_path(
                    hypothesis.initial_state, state
                )
            if state.prefix not in self.freq_dict.keys():
                self.freq_dict[state.prefix] = 0

            states_to_cover.extend(
                [state] * (self.walks_per_state - self.freq_dict[state.prefix])
            )

        shuffle(states_to_cover)
        for state in states_to_cover:
            self.freq_dict[state.prefix] = self.freq_dict[state.prefix] + 1

            prefix = state.prefix
            random_walk = tuple(
                choice(self.alphabet) for _ in range(randint(1, self.random_walk_len))
            )

            test_case = prefix + random_walk + choice(hypothesis.characterization_set)
            cex = self.execute_test_case(hypothesis, test_case)
            if cex:
                return cex

        return None
