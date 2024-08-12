import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.vander)
class TorchVanderTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_vander_correctness(self):
        # Generate a random 1-D input tensor 
        tensor_size = random.randint(2,
                                     10)  # Size of the 1-D tensor should be at least 2 for meaningful Vandermonde matrix
        input_tensor = torch.tensor([random.uniform(0.1, 10.0) for _ in range(tensor_size)])

        # Optionally specify N
        if random.choice([True, False]):
            N = random.randint(1, tensor_size)
        else:
            N = None

        # Optionally toggle `increasing`
        increasing = random.choice([True, False])

        # Calculate vander matrix
        if N is not None:
            vander_matrix = torch.vander(input_tensor, N, increasing)
        else:
            vander_matrix = torch.vander(input_tensor)

        return vander_matrix
