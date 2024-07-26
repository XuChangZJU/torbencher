import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.distributions.half_normal.HalfNormal)
class TorchDistributionsHalfnormalHalfnormalTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_HalfNormal_correctness_small_scale(self):
        # Random dimension for the scale tensor
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Random input size
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Generate random scale tensor with small values
        scale = torch.rand(input_size) * 0.1  # scale is within (0, 0.1)
        # Create HalfNormal distribution
        m = torch.distributions.half_normal.HalfNormal(scale)
        # Sample from the distribution
        result = m.sample()
        return result
