
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.distributed.GradBucket)
class TorchGradBucketTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_grad_bucket_correctness(self):
        dim = random.randint(1, 10)
        buffer = torch.randn(dim)
        result = torch.distributed.GradBucket.pybind11_type(buffer)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_grad_bucket_large_scale(self):
        dim = random.randint(1000, 10000)
        buffer = torch.randn(dim)
        result = torch.distributed.GradBucket.pybind11_type(buffer)
        return result

