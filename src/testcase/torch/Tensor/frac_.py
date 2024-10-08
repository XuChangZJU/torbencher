import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.frac_)
class TorchTensorFracUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_frac__correctness(self):
        # Generate random dimension for the tensor
        dim = random.randint(1, 4)
        # Generate random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Generate input_size
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate random tensor 
        input_tensor = torch.randn(input_size)
        # In-place version of torch.Tensor.frac, no return value
        input_tensor.frac_()
        return input_tensor
