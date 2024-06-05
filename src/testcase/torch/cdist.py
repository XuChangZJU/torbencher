
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cdist)
class TorchCdistTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_cdist_correctness(self):
        dim1 = random.randint(1, 10)
        dim2 = random.randint(1, 10)
        tensor1 = torch.randn(dim1, 2)
        tensor2 = torch.randn(dim2, 2)
        result = torch.cdist(tensor1, tensor2)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_cdist_large_scale(self):
        dim1 = random.randint(100, 1000)
        dim2 = random.randint(100, 1000)
        tensor1 = torch.randn(dim1, 2)
        tensor2 = torch.randn(dim2, 2)
        result = torch.cdist(tensor1, tensor2)
        return result

