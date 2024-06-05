
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.optim.Optimizer)
class TorchOptimOptimizerTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_optimizer_correctness(self):
        dim = random.randint(1, 10)
        # Input is a random tensor with dimensions `dim`.
        input = torch.randn(dim)
        optimizer = torch.optim.Optimizer([input])
        optimizer.step()
        result = optimizer.state_dict()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_optimizer_large_scale(self):
        dim = random.randint(1000, 10000)
        # Input is a random tensor with dimensions `dim`.
        input = torch.randn(dim)
        optimizer = torch.optim.Optimizer([input])
        optimizer.step()
        result = optimizer.state_dict()
        return result

