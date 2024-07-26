import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.dot)
class TorchDotTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_dot_correctness(self):
        # `torch.dot` requires the input tensors to be 1D and have the same number of elements.
        dim = 1
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        input = torch.randn(input_size)
        other = torch.randn(input_size)
        result = torch.dot(input, other)
        return result
