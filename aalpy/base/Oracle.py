from abc import ABC, abstractmethod

from aalpy.base import SUL


class Oracle(ABC):
    """Abstract class implemented by all equivalence oracles."""

    def __init__(self, alphabet: list, sul: SUL):
        """
        Default constructor for all equivalence oracles.

        Args:

            alphabet: input alphabet
            sul: system under learning
        """

        self.alphabet = alphabet
        self.sul = sul
        self.num_queries = 0
        self.num_steps = 0

    @abstractmethod
    def find_cex(self, hypothesis):
        """
        Return a counterexample (inputs) that displays different behavior on system under learning and
        current hypothesis.

        Args:

          hypothesis: current hypothesis

        Returns:

            tuple or list containing counterexample inputs, None if no counterexample is found
        """
        pass

    def reset_hyp_and_sul(self, hypothesis):
        """
        Reset SUL and hypothesis to initial state.

        Args:

            hypothesis: current hypothesis

        """
        hypothesis.reset_to_initial()
        self.sul.post()
        self.sul.pre()
        self.num_queries += 1

    def execute_test_case(self, hypothesis, input):
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
