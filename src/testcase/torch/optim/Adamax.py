
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.optim.Adamax)
class TorchOptimAdamaxTestCase(TorBencherTestCaseBase):
    def test_adamax(self, input=None):
        if input is not None:
            result = torch.optim.Adamax(input[0], lr=input[1], betas=input[2], eps=input[3], weight_decay=input[4])
            return [result, input]
        params = [torch.randn(10, requires_grad=True), torch.randn(20, requires_grad=True)]
        lr = 1e-3
        betas = (0.9, 0.999)
        eps = 1e-8
        weight_decay = 0
        result = torch.optim.Adamax(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        return [result, [params, lr, betas, eps, weight_decay]]

