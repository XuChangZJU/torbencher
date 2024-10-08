import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.sigmoid)
class TorchSigmoidTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_sigmoid_correctness(self):
        # Define the dimensions of the input tensor
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate a random tensor of the specified size
        input_tensor = torch.randn(input_size)

        # Calculate the sigmoid of the input tensor
        result = torch.sigmoid(input_tensor)
        return result
