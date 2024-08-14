import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.bernoulli_)
class TorchTensorBernoulliUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_bernoulli__correctness(self):
        # Random dimension for the tensor
        dim = 4
        # Random number of elements each dimension
        num_of_elements_each_dim = 5
        # Random input size
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Generate a random tensor with integer dtype
        input_tensor = torch.randint(0, 10, input_size)
        # Generate a random probability between 0 and 1
        p = random.uniform(0, 1)
        # Apply bernoulli_ function
        result = input_tensor.bernoulli_(p)
        # Return the result tensor
        return result.shape
