import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.utils.rnn.pack_padded_sequence)
class TorchNnUtilsRnnPackUpaddedUsequenceTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_pack_padded_sequence_correctness(self):
        # Randomly generate input size
        dim = random.randint(2, 4)  # dim >= 2
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate random input tensor
        input_tensor = torch.randn(input_size)

        # Generate random lengths for each sequence in the batch
        batch_size = input_size[1] if len(input_size) >= 2 else 1
        lengths = sorted([random.randint(1, input_size[0]) for _ in range(batch_size)],
                         reverse=True)  # sorted lengths in decreasing order

        # Pack the padded sequence
        result = torch.nn.utils.rnn.pack_padded_sequence(input_tensor, lengths)

        return result
