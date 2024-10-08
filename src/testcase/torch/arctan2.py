import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.arctan2)
class TorchArctan2TestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_arctan2_correctness(self):
        # arctan2(input, other, *, out=None) -> Tensor
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        input = torch.randn(input_size)  # input tensor
        other = torch.randn(input_size)  # other tensor
        result = torch.arctan2(input, other)
        return result
