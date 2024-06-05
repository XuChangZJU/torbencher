
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.BoolTensor)
class TorchCudaBoolTensorTestCase(TorBencherTestCaseBase):
    def test_bool_tensor_correctness(self):
        dim = random.randint(1, 10)
        data = torch.randint(0, 2, (dim,))
        result = torch.cuda.BoolTensor(data)
        return result

    def test_bool_tensor_large_scale(self):
        dim = random.randint(1000, 10000)
        data = torch.randint(0, 2, (dim,))
        result = torch.cuda.BoolTensor(data)
        return result

