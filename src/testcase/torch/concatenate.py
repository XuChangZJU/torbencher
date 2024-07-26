import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.concatenate)
class TorchConcatenateTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_concatenate_correctness(self):
        # Randomly generate dimension for the tensors
        dim = random.randint(1, 4)
        # Randomly generate number of elements for each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Generate input_size based on dim and num_of_elements_each_dim
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Generate the number of tensors to concatenate
        num_of_tensors = random.randint(2, 5)
        # Generate a list of tensors to concatenate
        tensors = [torch.randn(input_size) for i in range(num_of_tensors)]
        # Randomly select an axis for concatenation, should be within the valid range
        axis = random.randint(0, len(input_size) - 1)
        result = torch.concatenate(tensors, axis)
        return result
