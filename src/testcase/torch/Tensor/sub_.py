import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.sub_)
class TorchTensorSubUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_sub__correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]

        input_tensor = torch.randn(input_size)
        other_tensor = torch.randn(input_size)  # other_tensor with the same size as input_tensor
        input_tensor.sub_(other_tensor)
        return input_tensor
