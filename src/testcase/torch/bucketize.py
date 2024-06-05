
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.bucketize)
class TorchBucketizeTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_bucketize_correctness(self):
        dim = random.randint(1, 10)
        input = torch.randn(dim)
        boundaries = torch.randn(dim)
        result = torch.bucketize(input, boundaries)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_bucketize_large_scale(self):
        dim = random.randint(1000, 10000)
        input = torch.randn(dim)
        boundaries = torch.randn(dim)
        result = torch.bucketize(input, boundaries)
        return result

