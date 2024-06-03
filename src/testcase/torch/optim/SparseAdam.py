
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.optim.SparseAdam)
class TorchOptimSparseAdamTestCase(TorBencherTestCaseBase):
    def test_sparse_adam(self, input=None):
        if input is not None:
            result = torch.optim.SparseAdam(input[0], lr=input[1], betas=input[2], eps=input[3])
            return result
        params = [torch.randn(10, requires_grad=True)]
        lr = 1e-3
        betas = (0.9, 0.999)
        eps = 1e-8
        result = torch.optim.SparseAdam(params, lr=lr, betas=betas, eps=eps)
        return result

