
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.optim.Rprop)
class TorchOptimRpropTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_rprop_correctness(self):
        dim = random.randint(1, 10)
        lr = random.uniform(0.01, 0.1)
        etas = (random.uniform(0.01, 0.1), random.uniform(0.01, 0.1))
        step_sizes = (random.uniform(0.01, 0.1), random.uniform(0.01, 0.1))
        # Input is a random tensor with dimensions `dim`.
        input = torch.randn(dim)
        optimizer = torch.optim.Rprop([input], lr=lr, etas=etas, step_sizes=step_sizes)
        optimizer.step()
        result = optimizer.state_dict()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_rprop_large_scale(self):
        dim = random.randint(1000, 10000)
        lr = random.uniform(0.01, 0.1)
        etas = (random.uniform(0.01, 0.1), random.uniform(0.01, 0.1))
        step_sizes = (random.uniform(0.01, 0.1), random.uniform(0.01, 0.1))
        # Input is a random tensor with dimensions `dim`.
        input = torch.randn(dim)
        optimizer = torch.optim.Rprop([input], lr=lr, etas=etas, step_sizes=step_sizes)
        optimizer.step()
        result = optimizer.state_dict()
        return result

