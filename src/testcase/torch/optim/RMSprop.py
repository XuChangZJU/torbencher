
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.optim.RMSprop)
class TorchOptimRMSpropTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_rmsprop_correctness(self):
        dim = random.randint(1, 10)
        lr = random.uniform(0.01, 0.1)
        alpha = random.uniform(0.01, 0.1)
        eps = random.uniform(0.01, 0.1)
        weight_decay = random.uniform(0.01, 0.1)
        momentum = random.uniform(0.01, 0.1)
        centered = random.choice([True, False])
        # Input is a random tensor with dimensions `dim`.
        input = torch.randn(dim)
        optimizer = torch.optim.RMSprop([input], lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay, momentum=momentum, centered=centered)
        optimizer.step()
        result = optimizer.state_dict()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_rmsprop_large_scale(self):
        dim = random.randint(1000, 10000)
        lr = random.uniform(0.01, 0.1)
        alpha = random.uniform(0.01, 0.1)
        eps = random.uniform(0.01, 0.1)
        weight_decay = random.uniform(0.01, 0.1)
        momentum = random.uniform(0.01, 0.1)
        centered = random.choice([True, False])
        # Input is a random tensor with dimensions `dim`.
        input = torch.randn(dim)
        optimizer = torch.optim.RMSprop([input], lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay, momentum=momentum, centered=centered)
        optimizer.step()
        result = optimizer.state_dict()
        return result

