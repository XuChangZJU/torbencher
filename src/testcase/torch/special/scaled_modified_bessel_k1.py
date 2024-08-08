import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.special.scaled_modified_bessel_k1)
class TorchSpecialScaledUmodifiedUbesselUk1TestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_scaled_modified_bessel_k1_correctness(self):
        # Randomly generate input tensor data
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
        input = torch.randn(input_size)  # input tensor

        result = torch.special.scaled_modified_bessel_k1(input)
        return result
