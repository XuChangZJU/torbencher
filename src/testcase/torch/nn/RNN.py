
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.RNN)
class TorchRNNTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_rnn_correctness(self):
        input_size = random.randint(1, 10)
        hidden_size = random.randint(1, 10)
        num_layers = random.randint(1, 10)
        input_tensor = torch.randn(random.randint(1, 10), random.randint(1, 10), input_size)
        rnn = torch.nn.RNN(input_size, hidden_size, num_layers=num_layers)
        result = rnn(input_tensor)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_rnn_large_scale(self):
        input_size = random.randint(100, 1000)
        hidden_size = random.randint(100, 1000)
        num_layers = random.randint(1, 10)
        input_tensor = torch.randn(random.randint(1000, 10000), random.randint(100, 1000), input_size)
        rnn = torch.nn.RNN(input_size, hidden_size, num_layers=num_layers)
        result = rnn(input_tensor)
        return result

