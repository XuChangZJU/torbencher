import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.lgamma_)
class TorchTensorLgammaUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_lgamma__correctness(self):
        # Randomly generate the input tensor size
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate a random tensor
        input_tensor = torch.randn(input_size)

        # Perform the lgamma_ operation
        input_tensor.lgamma_()

        # Return the result tensor
        return input_tensor
