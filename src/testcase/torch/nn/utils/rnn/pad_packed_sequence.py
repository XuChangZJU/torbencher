import random

import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.utils.rnn.pad_packed_sequence)
class TorchNnUtilsRnnPadUpackedUsequenceTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_pad_packed_sequence_correctness(self):
        # Randomly generate batch size and max sequence length
        batch_size = random.randint(1, 5)
        max_seq_len = random.randint(1, 10)

        # Generate random lengths for each sequence in the batch
        lengths = [random.randint(1, max_seq_len) for _ in range(batch_size)]

        # Create a random tensor with the shape (batch_size, max_seq_len)
        seq = torch.randn(batch_size, max_seq_len)

        # Pack the padded sequence
        packed_seq = pack_padded_sequence(seq, lengths, batch_first=True, enforce_sorted=False)

        # Unpack the packed sequence
        unpacked_seq, unpacked_lengths = pad_packed_sequence(packed_seq, batch_first=True)

        return unpacked_seq, unpacked_lengths
