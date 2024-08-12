import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.fft.ihfftn)
class TorchFftIhfftnTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_ihfftn_correctness(self):
        # Generate random dimension for the input tensor
        dim = random.randint(1, 4)
        # Generate random size for each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Create input size list
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Generate random real-valued input tensor
        input_tensor = torch.randn(input_size)
        # Generate random signal size in the transformed dimensions
        s = [random.randint(1, 10) for _ in range(dim)]
        # Perform ihfftn operation
        result = torch.fft.ihfftn(input_tensor, s)
        return result
