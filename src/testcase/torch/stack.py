
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.stack)
class TorchStackTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_stack_correctness(self):
        tensors = [torch.randn(random.randint(1, 10), random.randint(1, 10)) for _ in range(random.randint(1, 10))]
        dim = random.randint(0, 2)
        result = torch.stack(tensors, dim)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_stack_large_scale(self):
        tensors = [torch.randn(random.randint(1000, 10000), random.randint(1000, 10000)) for _ in range(random.randint(1000, 10000))]
        dim = random.randint(0, 2)
        result = torch.stack(tensors, dim)
        return result

