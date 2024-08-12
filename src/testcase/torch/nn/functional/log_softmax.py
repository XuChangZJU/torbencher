import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.functional.log_softmax)
class TorchNnFunctionalLogUsoftmaxTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_log_softmax_correctness(self):
        # Randomly generate the dimension of the input tensor
        dim = random.randint(1, 4)
        # Randomly generate the number of elements for each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Generate the input size for the tensor
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Generate a random tensor with the specified input size
        input_tensor = torch.randn(input_size)
        # Randomly select a dimension along which to compute the log_softmax
        dim_to_compute = random.randint(0, len(input_size) - 1)
        # Apply the log_softmax operation
        result = torch.nn.functional.log_softmax(input_tensor, dim_to_compute)
        return result
