import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.map_)
class TorchTensorMapUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_map__correctness(self):
        # Define the dimensions of the tensors
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Create random tensors with the specified dimensions
        tensor1 = torch.randn(input_size)
        tensor2 = torch.randn(input_size)  # tensor2 needs to be broadcastable with tensor1

        # Define a simple callable function
        def callable_function(a, b):
            return a + b

        # Apply the map_ function
        tensor1.map_(tensor2, callable_function)

        return tensor1
