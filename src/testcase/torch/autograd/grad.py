import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.autograd.grad)
class TorchAutogradGradTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_grad_correctness(self):
        # Random dimension for the tensors
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Random input size
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Random input tensor
        input_tensor = torch.randn(input_size, requires_grad=True)
        # Random output tensor, make sure it's differentiable from input_tensor
        output_tensor = torch.sum(input_tensor * torch.randn(input_size))
        # Compute gradients
        gradients = torch.autograd.grad(output_tensor, input_tensor)
        return gradients
