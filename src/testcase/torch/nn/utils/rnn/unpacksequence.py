import torch
import random
from torch.nn.utils.rnn import pack_sequence, unpack_sequence


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.utils.rnn.unpacksequence)
class TorchNnUtilsRnnUnpacksequenceTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_unpack_sequence_correctness(self):
    # Randomly generate the number of sequences
    num_sequences = random.randint(1, 5)
    
    # Randomly generate the length of each sequence
    sequences = [torch.randn(random.randint(1, 5)) for _ in range(num_sequences)]
    
    # Pack the sequences
    packed_sequences = pack_sequence(sequences)
    
    # Unpack the sequences
    unpacked_sequences = unpack_sequence(packed_sequences)
    
    return unpacked_sequences
