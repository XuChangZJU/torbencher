
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.optim.LBFGS)
class TorchOptimLBFGSTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_lbfgs_correctness(self):
        dim = random.randint(1, 10)
        lr = random.uniform(0.01, 0.1)
        max_iter = random.randint(1, 10)
        max_eval = random.randint(1, 10)
        tolerance_grad = random.uniform(0.01, 0.1)
        tolerance_change = random.uniform(0.01, 0.1)
        history_size = random.randint(1, 10)
        line_search_fn = random.choice(['strong_wolfe', 'armijo'])
        # Input is a random tensor with dimensions `dim`.
        input = torch.randn(dim)
        optimizer = torch.optim.LBFGS([input], lr=lr, max_iter=max_iter, max_eval=max_eval, tolerance_grad=tolerance_grad, tolerance_change=tolerance_change, history_size=history_size, line_search_fn=line_search_fn)
        optimizer.step()
        result = optimizer.state_dict()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_lbfgs_large_scale(self):
        dim = random.randint(1000, 10000)
        lr = random.uniform(0.01, 0.1)
        max_iter = random.randint(1, 10)
        max_eval = random.randint(1, 10)
        tolerance_grad = random.uniform(0.01, 0.1)
        tolerance_change = random.uniform(0.01, 0.1)
        history_size = random.randint(1, 10)
        line_search_fn = random.choice(['strong_wolfe', 'armijo'])
        # Input is a random tensor with dimensions `dim`.
        input = torch.randn(dim)
        optimizer = torch.optim.LBFGS([input], lr=lr, max_iter=max_iter, max_eval=max_eval, tolerance_grad=tolerance_grad, tolerance_change=tolerance_change, history_size=history_size, line_search_fn=line_search_fn)
        optimizer.step()
        result = optimizer.state_dict()
        return result

