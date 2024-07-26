import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.baddbmm_)
class TorchTensorBaddbmmTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_baddbmm__correctness(self):
        batch_size = random.randint(1, 4)  # Random batch size
        M = random.randint(1, 5)  # Random number of rows for matrices
        N = random.randint(1, 5)  # Random number of columns for matrices
        P = random.randint(1, 5)  # Random number of columns for second matrix

        # Random tensors with appropriate sizes
        self_tensor = torch.randn(batch_size, M, N)
        batch1 = torch.randn(batch_size, M, P)
        batch2 = torch.randn(batch_size, P, N)

        # Perform in-place baddbmm_ operation
        result = self_tensor.baddbmm_(batch1, batch2)
        return result
