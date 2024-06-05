
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.Linear)
class TorchLinearTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_linear_correctness(self):
        in_features = random.randint(1, 10)
        out_features = random.randint(1, 10)
        input_tensor = torch.randn(random.randint(1, 10), in_features)
        linear = torch.nn.Linear(in_features, out_features)
        result = linear(input_tensor)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_linear_large_scale(self):
        in_features = random.randint(100, 1000)
        out_features = random.randint(100, 1000)
        input_tensor = torch.randn(random.randint(1000, 10000), in_features)
        linear = torch.nn.Linear(in_features, out_features)
        result = linear(input_tensor)
        return result

