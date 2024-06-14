import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.utils.rnn.packsequence)
class TorchNnUtilsRnnPacksequenceTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_pack_sequence_correctness(self):
    # Random number of sequences
    num_sequences = random.randint(2, 5)
    
    # Generate random lengths for each sequence
    lengths = [random.randint(1, 5) for _ in range(num_sequences)]
    
    # Sort lengths in decreasing order
    lengths.sort(reverse=True)
    
    # Generate random sequences based on the lengths
    sequences = [torch.randn(length, random.randint(1, 3)) for length in lengths]
    
    # Pack the sequences
    packed_sequence = torch.nn.utils.rnn.pack_sequence(sequences)
    
    return packed_sequence
