import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.GRU)
class TorchNnGruTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_GRU_correctness(self):
        # Random parameters for GRU
        dim = random.randint(1, 4)
        batch_size = random.randint(1, 4)
        seq_len = random.randint(1, 10)
        input_size = random.randint(1, 10)
        hidden_size = random.randint(1, 10)
        num_layers = random.randint(1, 3)

        # Create input tensor
        input_tensor = torch.randn(seq_len, batch_size, input_size)

        # Create hidden state tensor
        h0 = torch.randn(num_layers, batch_size, hidden_size)

        # Create GRU module
        gru = torch.nn.GRU(input_size, hidden_size, num_layers)

        # Apply GRU
        output, hn = gru(input_tensor, h0)

        # Return the output and final hidden state
        return output, hn
