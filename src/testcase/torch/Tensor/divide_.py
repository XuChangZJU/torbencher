import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.divide_)
class TorchTensorDivideUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_divide__correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]

        input_tensor = torch.randn(input_size)
        value_tensor = torch.randn(input_size)  # The tensor to divide the input tensor by
        input_tensor.divide_(value_tensor)
        return input_tensor
