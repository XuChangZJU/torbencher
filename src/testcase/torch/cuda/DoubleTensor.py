
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.DoubleTensor)
class TorchCudaDoubleTensorTestCase(TorBencherTestCaseBase):
    def test_double_tensor_correctness(self):
        dim = random.randint(1, 10)
        data = torch.randn(dim, dtype=torch.double)
        result = torch.cuda.DoubleTensor(data)
        return result

    def test_double_tensor_large_scale(self):
        dim = random.randint(1000, 10000)
        data = torch.randn(dim, dtype=torch.double)
        result = torch.cuda.DoubleTensor(data)
        return result

