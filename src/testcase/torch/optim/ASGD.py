
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.optim.ASGD)
class TorchOptimASGDTestCase(TorBencherTestCaseBase):
    def test_asgd(self, input=None):
        if input is not None:
            result = torch.optim.ASGD(input[0], lr=input[1], lambd=input[2], alpha=input[3], t0=input[4], weight_decay=input[5])
            return [result, input]
        params = [torch.randn(10, requires_grad=True), torch.randn(20, requires_grad=True)]
        lr = 1e-3
        lambd = 1e-4
        alpha = 0.75
        t0 = 1e6
        weight_decay = 0
        result = torch.optim.ASGD(params, lr=lr, lambd=lambd, alpha=alpha, t0=t0, weight_decay=weight_decay)
        return [result, [params, lr, lambd, alpha, t0, weight_decay]]

