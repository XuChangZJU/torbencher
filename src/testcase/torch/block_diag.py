import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.block_diag)
class TorchBlockUdiagTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_block_diag_correctness(self):
        # Generate random dimensions for the tensors
        num_tensors = random.randint(1, 5)
        tensor_dims = [(random.randint(1, 5), random.randint(1, 5)) if random.random() < 0.5 else (
        random.randint(1, 5),) if random.random() < 0.5 else (random.randint(1, 5),) for _ in range(num_tensors)]

        # Create random tensors
        tensors = [torch.randn(*dim) for dim in tensor_dims]

        # Apply torch.block_diag
        result = torch.block_diag(*tensors)
        return result
