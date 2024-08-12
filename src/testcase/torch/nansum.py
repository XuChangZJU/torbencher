import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nansum)
class TorchNansumTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_nansum_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]

        input = torch.randn(input_size)
        input = torch.where(torch.rand(input_size) > 0.5, input, torch.nan)  # Randomly replace some elements with NaN
        result = torch.nansum(input)
        return result
