
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.optim.Adam)
class TorchOptimAdamTestCase(TorBencherTestCaseBase):
    def test_adam(self, input=None):
        if input is not None:
            result = torch.optim.Adam(input[0], lr=input[1], betas=input[2], eps=input[3], weight_decay=input[4], amsgrad=input[5])
            return [result, input]
        params = [torch.randn(10, requires_grad=True), torch.randn(20, requires_grad=True)]
        lr = 1e-3
        betas = (0.9, 0.999)
        eps = 1e-8
        weight_decay = 0
        amsgrad = False
        result = torch.optim.Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        return [result, [params, lr, betas, eps, weight_decay, amsgrad]]

