
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.addcmul)
class TorchAddcmulTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_addcmul_correctness(self):
        input = torch.randn(random.randint(1, 10))
        tensor1 = torch.randn(random.randint(1, 10))
        tensor2 = torch.randn(random.randint(1, 10))
        value = random.uniform(0.1, 10.0)
        result = torch.addcmul(input, tensor1, tensor2, value=value)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_addcmul_large_scale(self):
        input = torch.randn(random.randint(1000, 10000))
        tensor1 = torch.randn(random.randint(1000, 10000))
        tensor2 = torch.randn(random.randint(1000, 10000))
        value = random.uniform(0.1, 10.0)
        result = torch.addcmul(input, tensor1, tensor2, value=value)
        return result

