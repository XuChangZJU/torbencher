import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.select)
class TorchTensorSelectTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_select_correctness(self):
        # Randomly generate tensor dimension and size
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate random tensor 
        input_tensor = torch.randn(input_size)
        # Generate random dim and index, making sure index is within the range of the tensor's dimension size
        dim = random.randint(0, len(input_size) - 1)
        index = random.randint(0, input_size[dim] - 1)
        result = input_tensor.select(dim, index)
        return result
