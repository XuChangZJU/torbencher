import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.fft.rfftn)
class TorchFftRfftnTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_rfftn_correctness(self):
        # Generate random dimension for the input tensor
        dim = random.randint(1, 4)
        # Generate random number of elements for each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Create input_size list for the input tensor
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Generate random input tensor
        input_tensor = torch.randn(input_size)
        # Calculate rfftn
        result = torch.fft.rfftn(input_tensor)
        return result
