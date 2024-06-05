
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.autograd.is_tensor_like)
class TorchAutogradIsTensorLikeTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_is_tensor_like_correctness(self):
        input = random.choice([torch.randn(random.randint(1, 10)), torch.tensor(random.randint(1, 10)), 1, [1, 2, 3], (1, 2, 3)])
        result = torch.autograd.is_tensor_like(input)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_is_tensor_like_large_scale(self):
        input = random.choice([torch.randn(random.randint(1000, 10000)), torch.tensor(random.randint(1000, 10000)), 1, [1, 2, 3], (1, 2, 3)])
        result = torch.autograd.is_tensor_like(input)
        return result


