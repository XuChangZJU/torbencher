import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.utils.rnn.unpad_sequence)
class TorchNnUtilsRnnUnpadUsequenceTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_unpad_sequence_correctness(self):
        from torch.nn.utils.rnn import pad_sequence, unpad_sequence
        a = torch.ones(25, 300)
        b = torch.ones(22, 300)
        c = torch.ones(15, 300)
        sequences = [a, b, c]
        padded_sequences = pad_sequence(sequences)
        lengths = torch.as_tensor([v.size(0) for v in sequences])
        unpadded_sequences = unpad_sequence(padded_sequences, lengths)
        return torch.allclose(sequences[0], unpadded_sequences[0]) and \
            torch.allclose(sequences[1], unpadded_sequences[1]) and \
            torch.allclose(sequences[2], unpadded_sequences[2])
