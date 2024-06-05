
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.CharTensor)
class TorchCharTensorTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_chartensor_correctness(self):
        dim = random.randint(1, 10)
        result = torch.randint(0, 256, (dim,))
        return result

    @test_api_version.larger_than("1.1.3")
    def test_chartensor_large_scale(self):
        dim = random.randint(1000, 10000)
        result = torch.randint(0, 256, (dim,))
        return result

