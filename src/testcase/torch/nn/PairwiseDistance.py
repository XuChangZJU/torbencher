import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.PairwiseDistance)
class TorchNnPairwisedistanceTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_pairwise_distance_correctness(self):
        # Randomly choose the batch dimension (N) and vector dimension (D)
        batch_dim = random.randint(1, 10)
        vector_dim = random.randint(1, 10)

        # Generate random tensors for input1 and input2 with shape (N, D)
        input1 = torch.randn(batch_dim, vector_dim)
        input2 = torch.randn(batch_dim, vector_dim)

        # Randomly choose the norm degree p
        p = random.uniform(-2.0, 2.0)

        # Create the PairwiseDistance module
        pdist = torch.nn.PairwiseDistance(p=p)

        # Compute the pairwise distance
        result = pdist(input1, input2)

        return result
