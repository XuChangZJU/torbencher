import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.autograd.gradcheck.gradgradcheck)
class TorchAutogradGradcheckGradgradcheckTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_gradgradcheck_correctness(self):
        # Random dimension for the tensor
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]

        # Random tensor with requires_grad=True
        tensor = torch.randn(input_size, requires_grad=True)

        # Define a simple function to test gradgradcheck
        def simple_function(x):
            return x ** 2

        # Perform gradgradcheck
        result = torch.autograd.gradgradcheck(simple_function, (tensor,))
        return result
