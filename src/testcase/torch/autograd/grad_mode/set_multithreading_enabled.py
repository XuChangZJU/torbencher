import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.autograd.grad_mode.set_multithreading_enabled)
class TorchAutogradGradUmodeSetUmultithreadingUenabledTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_set_multithreading_enabled_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]

        tensor1 = torch.randn(input_size, requires_grad=True)
        tensor2 = torch.randn(input_size, requires_grad=True)
        mode = random.choice([True, False])  # Random mode selection

        with torch.autograd.grad_mode.set_multithreading_enabled(mode):
            output = torch.sum(tensor1 + tensor2)
            grad = torch.autograd.grad(output, [tensor1, tensor2])

        return grad
