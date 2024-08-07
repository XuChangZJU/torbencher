import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.utils.rnn.unpad_sequence)
class TorchNnUtilsRnnUnpadUsequenceTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_unpad_sequence_correctness(self):
        # Randomly generate the number of sequences.
        num_sequences = random.randint(1, 10)
        # Randomly generate the maximum sequence length.
        max_sequence_length = random.randint(1, 100)
        # Randomly generate the dimension of each element in the sequence.
        element_dim = random.randint(1, 100)

        # Generate random sequence lengths.
        sequence_lengths = [random.randint(1, max_sequence_length) for _ in range(num_sequences)]

        # Generate random sequences.
        sequences = [torch.randn(length, element_dim) for length in sequence_lengths]

        # Pad the sequences.
        padded_sequences = torch.nn.utils.rnn.pad_sequence(sequences)

        # Convert sequence lengths to a tensor.
        lengths_tensor = torch.as_tensor(sequence_lengths)

        # Unpad the sequences.
        unpadded_sequences = torch.nn.utils.rnn.unpad_sequence(padded_sequences, lengths_tensor)

        return unpadded_sequences
