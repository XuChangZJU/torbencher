import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.fft.irfftn)
class TorchFftIrfftnTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_irfftn_correctness(self):
        # Random dimension for the tensors
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Random input size
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Generate random tensor with the specified input size
        input_tensor = torch.randn(input_size)
        # Perform rfftn on the input tensor
        rfftn_result = torch.fft.rfftn(input_tensor)
        # Perform irfftn on the rfftn_result
        irfftn_result = torch.fft.irfftn(rfftn_result, s=input_size)
        # Return the irfftn_result
        return irfftn_result
