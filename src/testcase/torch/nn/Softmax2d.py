import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.Softmax2d)
class TorchNnSoftmax2dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_softmax2d_correctness(self):
        # Random input shape
        batch_size = random.randint(1, 4)
        num_channels = random.randint(1, 5)
        height = random.randint(1, 10)
        width = random.randint(1, 10)
        input_size = [batch_size, num_channels, height, width]

        # Generate random input tensor
        input_tensor = torch.randn(input_size)

        # Create Softmax2d instance
        softmax2d = torch.nn.Softmax2d()

        # Apply softmax2d to the input
        result = softmax2d(input_tensor)
        return result
