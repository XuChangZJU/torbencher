import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.functional.pairwise_distance)
class TorchNnFunctionalPairwiseUdistanceTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_pairwise_distance_correctness(self):
        # Random dimension for the tensors
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Random input size
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Random tensor 1
        x1 = torch.randn(input_size)
        # Random tensor 2 with the same size as tensor 1
        x2 = torch.randn(input_size)
        # Calculate pairwise distance
        result = torch.nn.functional.pairwise_distance(x1, x2)
        return result
