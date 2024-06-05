
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.allclose)
class TorchAllcloseTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_allclose_correctness(self):
        dim = random.randint(1, 10)
        tensor1 = torch.randn(dim)
        tensor2 = torch.randn(dim)
        rtol = random.uniform(0.1, 10.0)
        atol = random.uniform(0.1, 10.0)
        result = torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_allclose_large_scale(self):
        dim = random.randint(1000, 10000)
        tensor1 = torch.randn(dim)
        tensor2 = torch.randn(dim)
        rtol = random.uniform(0.1, 10.0)
        atol = random.uniform(0.1, 10.0)
        result = torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol)
        return result

