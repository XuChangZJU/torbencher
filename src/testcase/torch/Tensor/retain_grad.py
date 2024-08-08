import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.retain_grad)
class TorchTensorRetainUgradTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_retain_grad_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]

        tensor = torch.randn(input_size, requires_grad=True)  # Tensor with requires_grad=True
        intermediate_tensor = tensor * 2  # Intermediate tensor
        intermediate_tensor.retain_grad()  # Enable grad retention for the intermediate tensor
        output_tensor = intermediate_tensor.mean()  # Output tensor
        output_tensor.backward()  # Backward pass
        return tensor.grad, intermediate_tensor.grad  # Return gradients of both tensors
