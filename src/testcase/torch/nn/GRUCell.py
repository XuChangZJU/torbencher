
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.GRUCell)
class TorchGRUCellTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_grucell_correctness(self):
        input_size = random.randint(1, 10)
        hidden_size = random.randint(1, 10)
        input_tensor = torch.randn(random.randint(1, 10), input_size)
        hidden = torch.randn(random.randint(1, 10), hidden_size)
        gru_cell = torch.nn.GRUCell(input_size, hidden_size)
        result = gru_cell(input_tensor, hidden)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_grucell_large_scale(self):
        input_size = random.randint(100, 1000)
        hidden_size = random.randint(100, 1000)
        input_tensor = torch.randn(random.randint(1000, 10000), input_size)
        hidden = torch.randn(random.randint(1000, 10000), hidden_size)
        gru_cell = torch.nn.GRUCell(input_size, hidden_size)
        result = gru_cell(input_tensor, hidden)
        return result

