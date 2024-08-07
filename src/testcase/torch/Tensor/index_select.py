import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.index_select)
class TorchTensorIndexUselectTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_index_select_correctness(self):
        # Randomly generate tensor dimension and size
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Create random tensor 
        input_tensor = torch.randn(input_size)

        # Randomly select dimension
        dim = random.randint(0, len(input_size) - 1)

        # Generate valid random indices
        index_size = random.randint(1, input_size[dim])
        index = torch.randint(0, input_size[dim], (index_size,))

        # Perform index_select operation
        result = input_tensor.index_select(dim, index)
        return result
