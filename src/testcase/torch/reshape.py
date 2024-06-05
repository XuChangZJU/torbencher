
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.reshape)
class TorchReshapeTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_reshape_correctness(self):
        input = torch.randn(random.randint(1, 10), random.randint(1, 10), random.randint(1, 10))
        shape = (random.randint(1, 10), random.randint(1, 10), random.randint(1, 10))
        result = torch.reshape(input, shape)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_reshape_large_scale(self):
        input = torch.randn(random.randint(1000, 10000), random.randint(1000, 10000), random.randint(1000, 10000))
        shape = (random.randint(1000, 10000), random.randint(1000, 10000), random.randint(1000, 10000))
        result = torch.reshape(input, shape)
        return result

