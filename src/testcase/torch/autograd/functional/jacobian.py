import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.autograd.functional.jacobian)
class TorchAutogradFunctionalJacobianTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_jacobian_correctness(self):
        # Define the function for which to compute the Jacobian
        def func(x, y):
            return x ** 2 + 2 * x * y + y ** 2, x + y

        # Generate random input tensors
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
        x = torch.randn(input_size, requires_grad=True)
        y = torch.randn(input_size, requires_grad=True)

        # Compute the Jacobian
        jacobian = torch.autograd.functional.jacobian(func, (x, y))

        return jacobian
