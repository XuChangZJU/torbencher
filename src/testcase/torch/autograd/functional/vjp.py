import torch
import random
from torch.autograd.functional import vjp

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.autograd.functional.vjp)
class TorchAutogradFunctionalVjpTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_vjp_correctness(self):
        # Define the function for vjp
        def func(x, y):
            return x ** 2 + 2 * y

        # Generate random input tensors
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
        x = torch.randn(input_size, requires_grad=True)
        y = torch.randn(input_size, requires_grad=True)

        # Generate random vector v
        v = torch.randn(input_size)

        # Compute vjp
        output, vjp_result = vjp(func, (x, y), v)

        return output, vjp_result
