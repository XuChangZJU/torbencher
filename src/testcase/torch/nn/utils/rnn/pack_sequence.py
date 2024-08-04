import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.utils.rnn.pack_sequence)
class TorchNnUtilsRnnPacksequenceTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_pack_sequence_correctness(self):
        # Random number of sequences
        num_sequences = random.randint(2, 5)

        # Generate random lengths for each sequence
        lengths = [random.randint(1, 5) for _ in range(num_sequences)]

        # Sort lengths in decreasing order
        lengths.sort(reverse=True)

        # lengths = lengths.cpu()

        # Generate random sequences based on the lengths
        sequences = [torch.randn(length, random.randint(1, 3)) for length in lengths]
        # for sequence in sequences:
        #     sequence = sequence.to(torch.device("cpu"))
        # Pack the sequences
        # packed_sequence = torch.nn.utils.rnn.pack_sequence(sequences)

        lengths = torch.as_tensor([v.size(0) for v in sequences])
        return torch.nn.utils.rnn.pack_padded_sequence(torch.nn.utils.rnn.pad_sequence(sequences), lengths.cpu())

        # return packed_sequence
