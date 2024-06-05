
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.ShortTensor)
class TorchCudaShortTensorTestCase(TorBencherTestCaseBase):
    def test_short_tensor_correctness(self):
        dim = random.randint(1, 10)
        data = torch.randint(-100, 100, (dim,))
        result = torch.cuda.ShortTensor(data)
        return result

    def test_short_tensor_large_scale(self):
        dim = random.randint(1000, 10000)
        data = torch.randint(-100, 100, (dim,))
        result = torch.cuda.ShortTensor(data)
        return result

