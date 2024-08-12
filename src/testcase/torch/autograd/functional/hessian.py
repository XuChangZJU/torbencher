import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.autograd.functional.hessian)
class TorchAutogradFunctionalHessianTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_hessian_correctness(self):
        """
        Test the correctness of the hessian function with small scale random parameters.
        """
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Define a simple function for testing
        def func(x, y):
            return (x * y).sum()

        # Generate random input tensors
        x = torch.randn(input_size, requires_grad=True)
        y = torch.randn(input_size, requires_grad=True)

        # Compute the Hessian using torch.autograd.functional.hessian
        hessian_result = torch.autograd.functional.hessian(func, (x, y))

        # Return the Hessian result
        return hessian_result
