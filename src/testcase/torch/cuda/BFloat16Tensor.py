
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.BFloat16Tensor)
class TorchCudaBFloat16TensorTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.10.0")
    def test_bfloat16_tensor_correctness(self):
        dim = random.randint(1, 10)
        data = torch.randn(dim, dtype=torch.bfloat16)
        result = torch.cuda.BFloat16Tensor(data)
        return result

    @test_api_version.larger_than("1.10.0")
    def test_bfloat16_tensor_large_scale(self):
        dim = random.randint(1000, 10000)
        data = torch.randn(dim, dtype=torch.bfloat16)
        result = torch.cuda.BFloat16Tensor(data)
        return result

