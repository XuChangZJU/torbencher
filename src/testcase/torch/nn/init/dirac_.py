import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.init.dirac_)
class TorchNnInitDiracUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_dirac_correctness(self):
        # Random dimension for the tensors
        dim = random.randint(3, 5)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # groups should be a divisor of the number of input channels
        groups = random.choice([i for i in range(1, input_size[0] + 1) if input_size[0] % i == 0])
        tensor = torch.randn(input_size)
        result = torch.nn.init.dirac_(tensor, groups)
        return result
