import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.sin_)
class TorchTensorSinUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_sin__correctness(self):
        """
        Test the correctness of torch.Tensor.sin_()
        """
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        input_tensor = torch.randn(input_size)
        input_tensor.sin_()
        return input_tensor
