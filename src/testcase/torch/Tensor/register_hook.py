import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.register_hook)
class TorchTensorRegisterUhookTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_register_hook_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]

        tensor = torch.randn(input_size, requires_grad=True)
        grad_multiplier = random.uniform(0.1, 10.0)  # Random gradient multiplier

        # Define a hook that multiplies the gradient by grad_multiplier
        def hook_fn(grad):
            return grad * grad_multiplier

        handle = tensor.register_hook(hook_fn)

        # Compute gradients
        tensor.sum().backward()

        # Retrieve the modified gradient
        result = tensor.grad

        handle.remove()  # Remove the hook

        return result
