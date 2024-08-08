import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.baddbmm)
class TorchTensorBaddbmmTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_baddbmm_correctness(self):
        batch_size = random.randint(1, 4)  # Random batch size
        m = random.randint(1, 5)  # Random number of rows for matrices in batch1
        n = random.randint(1, 5)  # Random number of columns for matrices in batch2
        p = random.randint(1, 5)  # Random number of columns for matrices in batch1 and rows for matrices in batch2

        # Random tensors for batch1 and batch2
        batch1 = torch.randn(batch_size, m, p)
        batch2 = torch.randn(batch_size, p, n)
        input_tensor = torch.randn(batch_size, m, n)  # Random input tensor

        # Perform baddbmm operation
        result = input_tensor.baddbmm(batch1, batch2)
        return result
