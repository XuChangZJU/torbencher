import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.Dropout2d)
class TorchNnDropout2dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_dropout2d_correctness(self):
    # Random input size
    batch_size = random.randint(1, 10)
    num_channels = random.randint(1, 16)
    height = random.randint(16, 32)
    width = random.randint(16, 32)
    input_size = (batch_size, num_channels, height, width)

    # Random input tensor
    input_tensor = torch.randn(input_size)

    # Random dropout probability
    p = random.uniform(0.1, 0.9)  # Probability between 0.1 and 0.9

    # Create a Dropout2d module
    dropout2d = torch.nn.Dropout2d(p)

    # Apply dropout
    output_tensor = dropout2d(input_tensor)

    return output_tensor
