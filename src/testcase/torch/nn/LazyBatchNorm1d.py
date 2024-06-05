
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.LazyBatchNorm1d)
class TorchLazyBatchNorm1dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_lazybatchnorm1d_correctness(self):
        num_features = random.randint(1, 10)
        input_tensor = torch.randn(random.randint(1, 10), num_features)
        lazy_batch_norm = torch.nn.LazyBatchNorm1d(num_features)
        result = lazy_batch_norm(input_tensor)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_lazybatchnorm1d_large_scale(self):
        num_features = random.randint(100, 1000)
        input_tensor = torch.randn(random.randint(1000, 10000), num_features)
        lazy_batch_norm = torch.nn.LazyBatchNorm1d(num_features)
        result = lazy_batch_norm(input_tensor)
        return result

