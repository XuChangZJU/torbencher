import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.autograd.functional.hvp)
class TorchAutogradFunctionalHvpTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_hvp_correctness(self):
        # Define the function for which to compute the Hessian vector product
        def pow_reducer(x):
            return x.pow(3).sum()

        # Generate random input tensor
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
        inputs = torch.randn(input_size)

        # Generate random vector v with the same size as inputs
        v = torch.randn(input_size)

        # Compute the Hessian vector product
        result = torch.autograd.functional.hvp(pow_reducer, inputs, v)
        return result
