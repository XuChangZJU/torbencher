import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.squeeze)
class TorchTensorSqueezeTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_squeeze_correctness(self):
        # Randomly generate tensor dimension and size
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate random tensor with some dimensions of size 1
        input_size[random.randint(0, dim - 1)] = 1  # Ensure at least one dimension has size 1
        tensor = torch.randn(input_size)

        result = tensor.squeeze()
        return result
