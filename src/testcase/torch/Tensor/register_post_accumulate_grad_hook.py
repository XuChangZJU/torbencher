import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.register_post_accumulate_grad_hook)
class TorchTensorRegisterpostaccumulategradhookTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_register_post_accumulate_grad_hook_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]

        tensor = torch.randn(input_size, requires_grad=True)  # Leaf tensor
        lr = random.uniform(0.01, 0.1)  # Random learning rate

        # Define a simple hook that subtracts lr * grad from the tensor
        def hook_fn(param):
            param.add_(param.grad, alpha=-lr)

        handle = tensor.register_post_accumulate_grad_hook(hook_fn)
        grad_output = torch.randn(input_size)
        tensor.backward(grad_output)

        return tensor
