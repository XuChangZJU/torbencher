import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api
import unittest

@test_api(torch.nn.Dropout1d)
class TorchNnDropout1dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    @unittest.skip
    def test_dropout1d_correctness(self):
        # Random input size
        batch_size = random.randint(1, 10)
        channels = random.randint(1, 10)
        length = random.randint(1, 10)
        input_size = [batch_size, channels, length]

        # Random input tensor
        input_tensor = torch.randn(input_size)

        # Random dropout probability
        p = random.uniform(0.1, 0.9)  # Probability between 0.1 and 0.9

        # Create Dropout1d module
        dropout = torch.nn.Dropout1d(p)

        # Apply dropout
        result = dropout(input_tensor)

        return result
