import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.log_)
class TorchTensorLogUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_log__correctness(self):
        # Generate random dimension and size for the tensor
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate a random tensor with positive values to ensure log is valid
        tensor = torch.randn(input_size).abs_()
        result = tensor.log_()
        return result
