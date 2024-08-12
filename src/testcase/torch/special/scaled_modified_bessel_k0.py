import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.special.scaled_modified_bessel_k0)
class TorchSpecialScaledUmodifiedUbesselUk0TestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_scaled_modified_bessel_k0_correctness(self):
        # Generate random dimension for the input tensor
        dim = random.randint(1, 4)
        # Generate random number of elements for each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Generate input size
        input_size = [num_of_elements_each_dim for _ in range(dim)]
        # Generate random input tensor
        input_tensor = torch.randn(input_size)
        input_tensor = torch.abs(input_tensor) + 1
        # Calculate scaled modified Bessel function of the second kind of order 0
        result = torch.special.scaled_modified_bessel_k0(input_tensor)
        return result
