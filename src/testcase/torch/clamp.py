import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.clamp)
class TorchClampTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_clamp_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]

        input_tensor = torch.randn(input_size)
        min_value = random.uniform(-10.0, 0.0)  # Random min value between -10.0 and 0.0
        max_value = random.uniform(0.0, 10.0)  # Random max value between 0.0 and 10.0
        result = torch.clamp(input_tensor, min_value, max_value)
        return result
