import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api
import unittest

@test_api(torch.nn.RNN)
class TorchNnRnnTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    # @unittest.skip
    def test_rnn_correctness(self):
        # Random parameters for RNN
        input_size = random.randint(1, 10)  # Random input size
        hidden_size = random.randint(1, 10)  # Random hidden size
        num_layers = random.randint(1, 3)  # Random number of layers
        seq_len = random.randint(5, 10)  # Random sequence length
        batch_size = random.randint(1, 3)  # Random batch size

        # Create input tensor
        input_tensor = torch.randn(seq_len, batch_size, input_size)

        # Create initial hidden state
        h0 = torch.randn(num_layers, batch_size, hidden_size)

        # Create RNN
        rnn = torch.nn.RNN(input_size, hidden_size, num_layers)

        # Randomly generate initialization parameters
        mean = random.uniform(-0.1, 0.1)
        std = random.uniform(0.01, 0.1)

        # Custom weight initialization
        with torch.no_grad():
            for name, param in rnn.named_parameters():
                if 'weight_ih' in name:
                    param.copy_(torch.normal(mean, std, param.size()))
                elif 'weight_hh' in name:
                    param.copy_(torch.normal(mean, std, param.size()))
                elif 'bias' in name:
                    param.copy_(torch.zeros(param.size()))  # Bias initialized to zero

        # Forward pass
        output, hn = rnn(input_tensor, h0)

        return output, hn
