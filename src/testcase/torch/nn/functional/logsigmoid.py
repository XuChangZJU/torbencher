import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.functional.logsigmoid)
class TorchNnFunctionalLogsigmoidTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_logsigmoid_correctness(self):
        # Randomly generate the dimension of the input tensor
        dim = random.randint(1, 4)
        # Randomly generate the number of elements for each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Create the input size list for the tensor
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Generate a random tensor with the specified input size
        input_tensor = torch.randn(input_size)
        # Calculate the logsigmoid of the input tensor
        result = torch.nn.functional.logsigmoid(input_tensor)
        # Return the result tensor
        return result
