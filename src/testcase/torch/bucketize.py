
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.bucketize)
class TorchBucketizeTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_bucketize_4d(self):
        a = torch.tensor([1, 2, 3])
        b = torch.tensor([0.5, 1.5, 2.5, 3.5])
        result = torch.bucketize(a, b)
        return result

