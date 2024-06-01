import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.NLLLoss)
class TorchNNNLLLossTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_nll_loss(self, input=None):
        if input is not None:
            result = torch.nn.NLLLoss()(input[0], input[1])
            return [result, input]
        a = torch.randn(3, 5, requires_grad=True)
        target = torch.empty(3, dtype=torch.long).random_(5)
        loss = torch.nn.NLLLoss()
        result = loss(a, target)
        return [result, [a, target]]
