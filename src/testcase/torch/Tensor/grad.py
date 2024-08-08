import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.grad)
class TorchTensorGradTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_grad_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]

        a = torch.randn(input_size, requires_grad=True)  # Tensor a, requires gradient calculation
        b = torch.randn(input_size, requires_grad=True)  # Tensor b, requires gradient calculation
        c = a + b
        d = c.sum()
        d.backward()  # Calculate gradients
        result = a.grad  # Get gradients of a
        return result
