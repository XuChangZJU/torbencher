import random
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(rnn_utils.pack_padded_sequence)
class TorchNnUtilsRnnPackUsequenceTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_pack_sequence_correctness(self):
        # Random number of sequences
        num_sequences = random.randint(2, 5)

        # Generate random lengths for each sequence
        lengths = [random.randint(1, 5) for _ in range(num_sequences)]

        # Sort lengths in decreasing order
        lengths.sort(reverse=True)

        # Set a fixed feature size for all sequences
        feature_size = random.randint(1, 3)

        # Generate random sequences based on the lengths and feature size
        sequences = [torch.randn(length, feature_size) for length in lengths]

        # Create a simple model with a Linear layer
        model = nn.Linear(feature_size, feature_size)

        # Initialize the weights and biases of the linear layer
        with torch.no_grad():
            model.weight = torch.nn.Parameter(torch.randn(model.out_features, model.in_features) * 0.01)
            if model.bias is not None:
                model.bias = torch.nn.Parameter(torch.randn(model.out_features) * 0.01)

        # Pack the sequences
        lengths = torch.as_tensor([v.size(0) for v in sequences])
        padded_sequences = rnn_utils.pad_sequence(sequences)
        packed_sequence = rnn_utils.pack_padded_sequence(padded_sequences, lengths.cpu())

        return packed_sequence
