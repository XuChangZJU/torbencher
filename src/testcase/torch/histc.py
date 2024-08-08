import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.histc)
class TorchHistcTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_histc_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensor
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]

        input_tensor = torch.randn(input_size)
        bins = random.randint(1, 100)  # Random number of bins
        min_val = random.uniform(-10.0, 10.0)  # Random min value
        max_val = random.uniform(min_val, 10.0)  # Random max value (ensuring max_val >= min_val)
        result = torch.histc(input_tensor, bins, min_val, max_val)
        return result
