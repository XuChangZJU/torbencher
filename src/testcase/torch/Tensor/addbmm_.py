import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.addbmm_)
class TorchTensorAddbmmUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_addbmm__correctness(self):
        # Random dimensions for the batch tensors
        batch_size = random.randint(1, 4)
        M = random.randint(1, 5)
        N = random.randint(1, 5)
        P = random.randint(1, 5)

        # Random tensors for batch1 and batch2
        batch1 = torch.randn(batch_size, M, N)
        batch2 = torch.randn(batch_size, N, P)

        # Random tensor for the input tensor
        input_tensor = torch.randn(M, P)

        # Perform the in-place addbmm_ operation
        result = input_tensor.addbmm_(batch1, batch2)

        return result
