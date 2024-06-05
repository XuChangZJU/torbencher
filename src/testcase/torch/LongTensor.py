
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.LongTensor)
class TorchLongTensorTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_longtensor_correctness(self):
        dim = random.randint(1, 10)
        result = torch.randint(-1000, 1000, (dim,))
        return result

    @test_api_version.larger_than("1.1.3")
    def test_longtensor_large_scale(self):
        dim = random.randint(1000, 10000)
        result = torch.randint(-1000, 1000, (dim,))
        return result

