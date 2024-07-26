import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.lerp)
class TorchLerpTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_lerp_correctness(self):
        # Random dimension for the tensors
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Random input size
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate random tensors with the same size
        start = torch.randn(input_size)
        end = torch.randn(input_size)
        # Generate random weight
        weight = random.uniform(0.1, 10.0)
        result = torch.lerp(start, end, weight)
        return result
