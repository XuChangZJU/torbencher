
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.autograd.cast)
class TorchAutogradCastTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_cast_correctness(self):
        input = torch.randn(random.randint(1, 10), requires_grad=True)
        dtype = random.choice([torch.float, torch.double, torch.int, torch.long])
        result = torch.autograd.cast(input, dtype=dtype)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_cast_large_scale(self):
        input = torch.randn(random.randint(1000, 10000), requires_grad=True)
        dtype = random.choice([torch.float, torch.double, torch.int, torch.long])
        result = torch.autograd.cast(input, dtype=dtype)
        return result


