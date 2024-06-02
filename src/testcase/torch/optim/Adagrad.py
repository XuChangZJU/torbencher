
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.optim.Adagrad)
class TorchOptimAdagradTestCase(TorBencherTestCaseBase):
    def test_adagrad_step(self, input=None):
        if input is not None:
            result = input[2].step()
            return [result, input]
        a = torch.randn(10, 5, requires_grad=True)
        b = torch.randn(10, 5)
        optimizer = torch.optim.Adagrad([a], lr=0.01)
        result = optimizer.step()
        return [result, [a, b, optimizer]]

