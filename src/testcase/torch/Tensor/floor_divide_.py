import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.floor_divide_)
class TorchTensorFloorUdivideUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_floor_divide__correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]

        input_tensor = torch.randn(input_size)
        value = random.uniform(0.1, 10.0)  # Random value between 0.1 and 10.0
        input_tensor.floor_divide_(value)
        return input_tensor
