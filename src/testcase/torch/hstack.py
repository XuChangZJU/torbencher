import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.hstack)
class TorchHstackTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_hstack_correctness(self):
        # Randomly choose the number of tensors to stack
        num_tensors = random.randint(2, 5)

        # Generate random dimensions for the tensors
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Make sure the first dimension (number of columns for 2D+, or number of elements for 1D) is the same for all tensors
        tensors = [torch.randn(*input_size) for _ in range(num_tensors)]

        result = torch.hstack(tensors)
        return result
