import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.BatchNorm2d)
class TorchNnBatchnorm2dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_batchnorm2d_correctness(self):
        # Randomly generate the number of features (channels)
        num_features = random.randint(2, 10)

        # Randomly generate the dimensions for the input tensor
        batch_size = random.randint(2, 5)
        height = random.randint(2, 10)
        width = random.randint(2, 10)

        # Create a random input tensor with the shape (N, C, H, W)
        input_tensor = torch.randn(batch_size, num_features, height, width)

        # Initialize BatchNorm2d with the randomly generated number of features
        batch_norm = torch.nn.BatchNorm2d(num_features)

        # Apply BatchNorm2d to the input tensor
        output_tensor = batch_norm(input_tensor)

        return output_tensor
