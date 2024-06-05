
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.as_tensor)
class TorchAsTensorTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_as_tensor_correctness(self):
        data = torch.randn(random.randint(1, 10))
        result = torch.as_tensor(data)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_as_tensor_large_scale(self):
        data = torch.randn(random.randint(1000, 10000))
        result = torch.as_tensor(data)
        return result

