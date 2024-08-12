import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.seed)
class TorchSeedTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_seed_correctness(self):
        # No input parameters for torch.seed
        seed1 = torch.seed()
        tensor1 = torch.randn(random.randint(1, 10))  # Generate a random tensor after setting the seed
        seed2 = torch.seed()
        tensor2 = torch.randn(random.randint(1, 10))  # Generate another random tensor after resetting the seed
        return seed1, seed2, tensor1, tensor2  # Return seeds and tensors to check if seeds are different and tensors are different
