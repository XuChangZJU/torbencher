
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.detach_copy)
class TorchDetachCopyTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_detach_copy_correctness(self):
        dim = random.randint(1, 10)
        tensor = torch.randn(dim, requires_grad=True)
        result = tensor.detach_copy()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_detach_copy_large_scale(self):
        dim = random.randint(1000, 10000)
        tensor = torch.randn(dim, requires_grad=True)
        result = tensor.detach_copy()
        return result

