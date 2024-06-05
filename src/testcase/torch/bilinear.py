
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.bilinear)
class TorchBilinearTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_bilinear_correctness(self):
        batch = random.randint(1, 10)
        in1_features = random.randint(1, 10)
        in2_features = random.randint(1, 10)
        out_features = random.randint(1, 10)
        input1 = torch.randn(batch, in1_features)
        input2 = torch.randn(batch, in2_features)
        weight = torch.randn(out_features, in1_features, in2_features)
        bias = torch.randn(out_features)
        result = torch.bilinear(input1, input2, weight, bias)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_bilinear_large_scale(self):
        batch = random.randint(100, 1000)
        in1_features = random.randint(100, 1000)
        in2_features = random.randint(100, 1000)
        out_features = random.randint(100, 1000)
        input1 = torch.randn(batch, in1_features)
        input2 = torch.randn(batch, in2_features)
        weight = torch.randn(out_features, in1_features, in2_features)
        bias = torch.randn(out_features)
        result = torch.bilinear(input1, input2, weight, bias)
        return result

