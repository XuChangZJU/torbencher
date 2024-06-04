
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.optim.Adadelta)
class TorchOptimAdadeltaTestCase(TorBencherTestCaseBase):
    def test_adadelta_step(self):
        
        a = torch.randn(10, 5, requires_grad=True)
        b = torch.randn(10, 5)
        optimizer = torch.optim.Adadelta([a], lr=0.01)
        result = optimizer.step()
        return result
