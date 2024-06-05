
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.LazyLinear)
class TorchLazyLinearTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_lazylinear_correctness(self):
        out_features = random.randint(1, 10)
        input_tensor = torch.randn(random.randint(1, 10), random.randint(1, 10))
        lazy_linear = torch.nn.LazyLinear(out_features)
        result = lazy_linear(input_tensor)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_lazylinear_large_scale(self):
        out_features = random.randint(100, 1000)
        input_tensor = torch.randn(random.randint(1000, 10000), random.randint(100, 1000))
        lazy_linear = torch.nn.LazyLinear(out_features)
        result = lazy_linear(input_tensor)
        return result

