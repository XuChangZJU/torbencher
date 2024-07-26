import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.select)
class TorchSelectTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_select_correctness(self):
        # Random dimension for the tensor
        dim = random.randint(2, 4)

        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(2, 5)

        # Generate an input tensor size list 
        input_size = [num_of_elements_each_dim for _ in range(dim)]

        # Create a random tensor
        input_tensor = torch.randn(input_size)

        # Choose a random dimension to select from
        select_dim = random.randint(0, dim - 1)

        # Choose a random index in the range of the selected dimension
        select_index = random.randint(0, input_size[select_dim] - 1)

        # Perform the torch.select operation
        result = torch.select(input_tensor, select_dim, select_index)

        return result
