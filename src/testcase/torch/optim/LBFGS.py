
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.optim.LBFGS)
class TorchOptimLBFGSTestCase(TorBencherTestCaseBase):
    def test_lbfgs(self):
        params = [torch.randn(10, requires_grad=True), torch.randn(20, requires_grad=True)]
        lr = 1
        max_iter = 20
        max_eval = None
        tolerance_grad = 1e-5
        tolerance_change = 1e-9
        history_size = 100
        line_search_fn = None
        result = torch.optim.LBFGS(params, lr=lr, max_iter=max_iter, max_eval=max_eval, tolerance_grad=tolerance_grad, tolerance_change=tolerance_change, history_size=history_size, line_search_fn=line_search_fn)
        return result
