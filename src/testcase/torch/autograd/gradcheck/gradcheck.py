import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.autograd.gradcheck.gradcheck)
class TorchAutogradGradcheckGradcheckTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_gradcheck_correctness(self):
        # Random dimension for the tensors
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]

        # Create a random tensor with requires_grad=True
        tensor = torch.randn(input_size, dtype=torch.double, requires_grad=True)

        # Define a simple function to test
        def simple_function(x):
            return x ** 2

        # Perform gradcheck
        result = torch.autograd.gradcheck(simple_function, tensor)
        return result
