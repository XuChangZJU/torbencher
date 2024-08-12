import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.SyncBatchNorm)
class TorchNnSyncbatchnormTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_sync_batch_norm_correctness(self):
        # Randomly generate the number of features (channels)
        num_features = random.randint(1, 10)

        # Randomly generate the dimensions of the input tensor
        N = random.randint(1, 4)  # Batch size
        D1 = random.randint(1, 5)  # Dimension 1
        D2 = random.randint(1, 5)  # Dimension 2
        D3 = random.randint(1, 5)  # Dimension 3

        # Create a random input tensor with the shape (N, num_features, D1, D2, D3)
        input_tensor = torch.randn(N, num_features, D1, D2, D3)

        # Create a SyncBatchNorm layer with the randomly generated number of features
        sync_batch_norm = torch.nn.SyncBatchNorm(num_features)

        # Apply the SyncBatchNorm layer to the input tensor
        output_tensor = sync_batch_norm(input_tensor)

        return output_tensor
