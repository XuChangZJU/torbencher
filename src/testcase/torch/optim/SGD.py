
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.optim.SGD)
class TorchOptimSGDTestCase(TorBencherTestCaseBase):
    def test_sgd(self):
        params = [torch.randn(10, requires_grad=True), torch.randn(20, requires_grad=True)]
        lr = 1e-3
        momentum = 0
        dampening = 0
        weight_decay = 0
        nesterov = False
        result = torch.optim.SGD(params, lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)
        return result

