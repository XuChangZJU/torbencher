import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.mvlgamma)
class TorchTensorMvlgammaTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_mvlgamma_correctness(self):
        # Random dimension for the tensor
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Generate random input size
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Generate random tensor with values greater than (p - 1) / 2, where p is the input argument to mvlgamma
        input_tensor = torch.randn(input_size) + torch.randint(1, 10, input_size) * 2
        # Random p value
        p = random.randint(1, input_tensor.size(dim - 1) * 2)
        # Calculate mvlgamma
        result = input_tensor.mvlgamma(p)
        return result
