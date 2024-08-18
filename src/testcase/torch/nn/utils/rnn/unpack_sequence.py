import random
import unittest
import torch
from torch.nn.utils.rnn import unpack_sequence, pack_sequence

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.utils.rnn.unpack_sequence)
class TorchNnUtilsRnnUnpackUsequenceTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_unpack_sequence_correctness(self):
        # Randomly generate the number of sequences
        num_sequences = random.randint(1, 5)

        # Randomly generate the length of each sequence
        lengths = [random.randint(1, 5) for _ in range(num_sequences)]

        # Set a fixed feature size for all sequences
        feature_size = random.randint(1, 3)

        # Generate random sequences based on the lengths and feature size
        sequences = [torch.randn(length, feature_size) for length in lengths]

        # Pack the sequences
        packed_sequences = pack_sequence(sequences, enforce_sorted=False)

        # Unpack the sequences
        unpacked_sequences = unpack_sequence(packed_sequences)

        return unpacked_sequences
