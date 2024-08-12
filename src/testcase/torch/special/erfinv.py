import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.special.erfinv)
class TorchSpecialErfinvTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_erfinv_correctness(self):
        # Random dimension for the tensor
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Generate random input size
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate random input tensor with values between -1 and 1
        input_tensor = torch.rand(input_size) * 2 - 1
        result = torch.special.erfinv(input_tensor)
        return result
