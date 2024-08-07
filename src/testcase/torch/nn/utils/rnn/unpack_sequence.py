import torch
import random
from torch.nn.utils.rnn import pack_sequence, unpack_sequence

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.utils.rnn.unpack_sequence)
class TorchNnUtilsRnnUnpackUsequenceTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_unpack_sequence_correctness(self):
        # Randomly generate the number of sequences
        num_sequences = random.randint(1, 5)

        # Randomly generate the length of each sequence
        sequences = [torch.randn(random.randint(1, 5)) for _ in range(num_sequences)]

        # Pack the sequences
        # packed_sequences = pack_sequence(sequences)

        lengths = torch.as_tensor([v.size(0) for v in sequences])
        packed_sequences = torch.nn.utils.rnn.pack_padded_sequence(torch.nn.utils.rnn.pad_sequence(sequences), lengths.cpu())

        # Unpack the sequences
        unpacked_sequences = unpack_sequence(packed_sequences)

        return unpacked_sequences
