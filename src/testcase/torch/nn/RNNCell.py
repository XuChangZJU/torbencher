
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.RNNCell)
class TorchNNRNNCellTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_rnn_cell(self):
        a = torch.randn(3, 10)
        hx = torch.randn(3, 20)
        rnn = torch.nn.RNNCell(10, 20)
        result = rnn(a, hx)
        return result

