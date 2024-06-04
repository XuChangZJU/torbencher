
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.optim.Rprop)
class TorchOptimRpropTestCase(TorBencherTestCaseBase):
    def test_rprop(self):
        
        params = [torch.randn(10, requires_grad=True), torch.randn(20, requires_grad=True)]
        lr = 1e-3
        etas = (0.5, 1.2)
        step_sizes = (1e-6, 50)
        result = torch.optim.Rprop(params, lr=lr, etas=etas, step_sizes=step_sizes)
        return result

