import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.requires_grad_)
class TorchTensorRequiresUgradUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_requires_grad__correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]

        tensor = torch.randn(input_size)
        requires_grad = bool(random.randint(0, 1))  # Random boolean for requires_grad
        result = tensor.requires_grad_(requires_grad)
        return result
