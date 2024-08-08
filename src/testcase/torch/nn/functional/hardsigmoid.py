import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.functional.hardsigmoid)
class TorchNnFunctionalHardsigmoidTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_hardsigmoid_correctness(self):
        # Define the dimension and size of the input tensor
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate a random tensor with values between -10 and 10
        input_tensor = torch.rand(input_size) * 20 - 10

        # Apply the hardsigmoid function
        result = torch.nn.functional.hardsigmoid(input_tensor)

        return result
