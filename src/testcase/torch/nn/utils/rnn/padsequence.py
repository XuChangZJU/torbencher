import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.utils.rnn.padsequence)
class TorchNnUtilsRnnPadsequenceTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_pad_sequence_correctness(self):
        # Randomly generate the number of sequences
        num_sequences = random.randint(2, 5)
        
        # Randomly generate the length of each sequence
        sequence_lengths = [random.randint(1, 10) for _ in range(num_sequences)]
        
        # Randomly generate the feature dimension
        feature_dim = random.randint(1, 100)
        
        # Create a list of random tensors with varying lengths
        sequences = [torch.randn(seq_len, feature_dim) for seq_len in sequence_lengths]
        
        # Apply pad_sequence
        padded_sequence = torch.nn.utils.rnn.pad_sequence(sequences)
        
        return padded_sequence
    