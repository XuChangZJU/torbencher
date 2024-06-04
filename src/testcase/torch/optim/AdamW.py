
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.optim.AdamW)
class TorchOptimAdamWTestCase(TorBencherTestCaseBase):
    def test_adamw(self):
        
        params = [torch.randn(10, requires_grad=True), torch.randn(20, requires_grad=True)]
        lr = 1e-3
        betas = (0.9, 0.999)
        eps = 1e-8
        weight_decay = 0.01
        amsgrad = False
        result = torch.optim.AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        return result

