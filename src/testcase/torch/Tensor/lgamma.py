import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.lgamma)
class TorchTensorLgammaTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_lgamma_correctness(self):
        """
        Test the correctness of lgamma() function in PyTorch.
        """
        dim = random.randint(1, 4)  # Random dimension for the tensor
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]

        input_tensor =torch.abs(torch.randn(input_size)) + 1e-6   # Generate random tensor
        result = input_tensor.lgamma()  # Calculate the lgamma of the tensor
        return result
