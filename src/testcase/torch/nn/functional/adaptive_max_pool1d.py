import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.functional.adaptive_max_pool1d)
class TorchNnFunctionalAdaptiveUmaxUpool1dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_adaptive_max_pool1d_correctness(self):
        # Random input size
        dim = random.randint(2, 3)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]
        input_size[1] = random.randint(1, 10)  # The length of the signal

        # Random input tensor
        input = torch.randn(input_size)

        # Random output size (single integer)
        output_size = random.randint(1, input_size[1])  # output size should be less than or equal to input size

        # Call adaptive_max_pool1d
        result = torch.nn.functional.adaptive_max_pool1d(input, output_size)
        return result
