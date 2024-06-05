
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.optim.ASGD)
class TorchOptimASGDTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_asgd_correctness(self):
        dim = random.randint(1, 10)
        lr = random.uniform(0.01, 0.1)
        lambd = random.uniform(0.01, 0.1)
        alpha = random.uniform(0.01, 0.1)
        t0 = random.randint(1, 10)
        weight_decay = random.uniform(0.01, 0.1)
        # Input is a random tensor with dimensions `dim`.
        input = torch.randn(dim)
        optimizer = torch.optim.ASGD([input], lr=lr, lambd=lambd, alpha=alpha, t0=t0, weight_decay=weight_decay)
        optimizer.step()
        result = optimizer.state_dict()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_asgd_large_scale(self):
        dim = random.randint(1000, 10000)
        lr = random.uniform(0.01, 0.1)
        lambd = random.uniform(0.01, 0.1)
        alpha = random.uniform(0.01, 0.1)
        t0 = random.randint(1, 10)
        weight_decay = random.uniform(0.01, 0.1)
        # Input is a random tensor with dimensions `dim`.
        input = torch.randn(dim)
        optimizer = torch.optim.ASGD([input], lr=lr, lambd=lambd, alpha=alpha, t0=t0, weight_decay=weight_decay)
        optimizer.step()
        result = optimizer.state_dict()
        return result

