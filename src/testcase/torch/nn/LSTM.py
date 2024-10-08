import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api
import unittest

@test_api(torch.nn.LSTM)
class TorchNnLstmTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    @unittest.skip
    def test_lstm_correctness(self):
        # Random parameters for LSTM
        input_size = random.randint(1, 10)  # Random input size
        hidden_size = random.randint(1, 10)  # Random hidden size
        num_layers = random.randint(1, 3)  # Random number of layers
        seq_len = random.randint(1, 5)  # Random sequence length
        batch_size = random.randint(1, 3)  # Random batch size

        # Create input tensor
        input_tensor = torch.randn(seq_len, batch_size, input_size)

        # Create LSTM model
        lstm = torch.nn.LSTM(input_size, hidden_size, num_layers)

        # Forward pass
        output, (hn, cn) = lstm(input_tensor)

        return output
