
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.autograd.functional.List)
class TorchAutogradFunctionalListTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.5.0")
    def test_list_correctness(self):
        dim = random.randint(1, 10)
        tensors = [torch.randn(dim, requires_grad=True) for _ in range(random.randint(1, 5))]
        result = torch.autograd.functional.List(tensors)
        return result

    @test_api_version.larger_than("1.5.0")
    def test_list_large_scale(self):
        dim = random.randint(1000, 10000)
        tensors = [torch.randn(dim, requires_grad=True) for _ in range(random.randint(1, 5))]
        result = torch.autograd.functional.List(tensors)
        return result

