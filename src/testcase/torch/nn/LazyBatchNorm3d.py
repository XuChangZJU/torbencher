import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.LazyBatchNorm3d)
class TorchNnLazybatchnorm3dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_lazy_batch_norm_3d_correctness(self):
        # Randomly generate dimensions for the 5D input tensor (N, C, D, H, W)
        N = random.randint(1, 4)  # Batch size
        C = random.randint(1, 4)  # Number of channels
        D = random.randint(2, 4)  # Depth
        H = random.randint(2, 4)  # Height
        W = random.randint(2, 4)  # Width

        # Generate random input tensor with the specified dimensions
        input_tensor = torch.randn(N, C, D, H, W)

        # Randomly generate parameters for LazyBatchNorm3d
        eps = random.uniform(1e-6, 1e-4)  # Epsilon for numerical stability
        momentum = random.uniform(0.05, 0.15)  # Momentum for running mean and variance

        # Initialize LazyBatchNorm3d with random parameters
        lazy_batch_norm_3d = torch.nn.LazyBatchNorm3d(eps, momentum)

        # Apply LazyBatchNorm3d to the input tensor
        result = lazy_batch_norm_3d(input_tensor)
        return result
