
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.dequantize)
class TorchDequantizeTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_dequantize_correctness(self):
        tensor = torch.randint(0, 10, (random.randint(1, 10),)).to(torch.quint8)
        result = torch.dequantize(tensor)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_dequantize_large_scale(self):
        tensor = torch.randint(0, 1000, (random.randint(1000, 10000),)).to(torch.quint8)
        result = torch.dequantize(tensor)
        return result

