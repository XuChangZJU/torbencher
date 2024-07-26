import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.baddbmm)
class TorchBaddbmmTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_baddbmm_correctness(self):
        # Randomly generate tensor dimensions
        b = random.randint(1, 10)  # Batch size
        n = random.randint(1, 10)
        m = random.randint(1, 10)
        p = random.randint(1, 10)

        # Generate input tensors with random data
        input_tensor = torch.randn(b, n, p)
        batch1 = torch.randn(b, n, m)
        batch2 = torch.randn(b, m, p)

        # Perform baddbmm operation
        result = torch.baddbmm(input_tensor, batch1, batch2)
        return result
