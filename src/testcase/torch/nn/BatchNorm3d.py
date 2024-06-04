
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.BatchNorm3d)
class TorchNNBatchNorm3dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_batch_norm3d(self):
        a = torch.randn(10, 100, 10, 10, 10)
        bn = torch.nn.BatchNorm3d(100)
        result = bn(a)
        return result

