import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.utils.rnn.PackedSequence)
class TorchNnUtilsRnnPackedsequenceTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_packedsequence_correctness(self):
    # Randomly generate input parameters for PackedSequence
    dim = random.randint(1, 4)
    num_of_elements_each_dim = random.randint(1, 5)
    input_size = [num_of_elements_each_dim for i in range(dim)]
    data = torch.randn(input_size)
    batch_sizes_length = random.randint(1, input_size[0])  # batch_sizes length should be less than or equal to data dim 0
    batch_sizes = torch.randint(1, input_size[0] + 1, (batch_sizes_length,))
    sorted_indices = torch.argsort(torch.randn(input_size[0]))
    unsorted_indices = torch.argsort(sorted_indices)

    # Create a PackedSequence object
    packed_sequence = torch.nn.utils.rnn.PackedSequence(data, batch_sizes, sorted_indices, unsorted_indices)

    return packed_sequence
