
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.optim.SparseAdam)
class TorchOptimSparseAdamTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_sparseadam_correctness(self):
        dim = random.randint(1, 10)
        lr = random.uniform(0.01, 0.1)
        betas = (random.uniform(0.01, 0.1), random.uniform(0.01, 0.1))
        eps = random.uniform(0.01, 0.1)
        # Input is a random tensor with dimensions `dim`.
        input = torch.randn(dim)
        optimizer = torch.optim.SparseAdam([input], lr=lr, betas=betas, eps=eps)
        optimizer.step()
        result = optimizer.state_dict()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_sparseadam_large_scale(self):
        dim = random.randint(1000, 10000)
        lr = random.uniform(0.01, 0.1)
        betas = (random.uniform(0.01, 0.1), random.uniform(0.01, 0.1))
        eps = random.uniform(0.01, 0.1)
        # Input is a random tensor with dimensions `dim`.
        input = torch.randn(dim)
        optimizer = torch.optim.SparseAdam([input], lr=lr, betas=betas, eps=eps)
        optimizer.step()
        result = optimizer.state_dict()
        return result

