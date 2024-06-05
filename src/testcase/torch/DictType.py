
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.DictType)
class TorchDictTypeTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_dicttype_correctness(self):
        dim = random.randint(1, 10)
        tensor1 = torch.randn(dim)
        tensor2 = torch.randn(dim)
        result = torch.DictType.ofTensors(tensor1, tensor2)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_dicttype_large_scale(self):
        dim = random.randint(1000, 10000)
        tensor1 = torch.randn(dim)
        tensor2 = torch.randn(dim)
        result = torch.DictType.ofTensors(tensor1, tensor2)
        return result

