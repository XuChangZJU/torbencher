import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.inner)
class TorchInnerTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_inner_correctness(self):
        # Test 1D tensors (dot product)
        dim = 1
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
        input = torch.randn(input_size)
        other = torch.randn(input_size)
        result = torch.inner(input, other)

        # Test multi-dimensional tensors
        dim = random.randint(2, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
        input_size[-1] = random.randint(1, 5)  # Ensure last dimension matches for input and other
        input = torch.randn(input_size)
        other_size = [num_of_elements_each_dim for i in range(dim)]
        other_size[-1] = input_size[-1]  # Ensure last dimension matches for input and other
        other = torch.randn(other_size)
        result = torch.inner(input, other)

        # Test scalar input
        input = torch.randn(input_size)
        other = torch.tensor(random.uniform(0.1, 10.0))  # Random scalar value
        result = torch.inner(input, other)
        return result
