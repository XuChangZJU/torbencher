import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.erfinv)
class TorchErfinvTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_erfinv_correctness(self):
        # The input value of torch.erfinv should be in the range [-1, 1].
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        input_tensor = torch.randn(input_size)  # Generate random tensor
        input_tensor = (input_tensor - input_tensor.min()) / (
                    input_tensor.max() - input_tensor.min()) * 2 - 1  # Normalize to [-1, 1]
        result = torch.erfinv(input_tensor)
        return result
