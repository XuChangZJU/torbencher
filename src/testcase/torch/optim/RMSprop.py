
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.optim.RMSprop)
class TorchOptimRMSpropTestCase(TorBencherTestCaseBase):
    def test_rmsprop(self, input=None):
        if input is not None:
            result = torch.optim.RMSprop(input[0], lr=input[1], alpha=input[2], eps=input[3], weight_decay=input[4], momentum=input[5], centered=input[6])
            return [result, input]
        params = [torch.randn(10, requires_grad=True), torch.randn(20, requires_grad=True)]
        lr = 1e-3
        alpha = 0.99
        eps = 1e-8
        weight_decay = 0
        momentum = 0
        centered = False
        result = torch.optim.RMSprop(params, lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay, momentum=momentum, centered=centered)
        return [result, [params, lr, alpha, eps, weight_decay, momentum, centered]]

