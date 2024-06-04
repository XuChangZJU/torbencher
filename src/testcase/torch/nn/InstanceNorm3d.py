
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.InstanceNorm3d)
class TorchNNInstanceNorm3dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_instance_norm3d(self):
        a = torch.randn(10, 20, 30, 40, 50)
        bn = torch.nn.InstanceNorm3d(20)
        result = bn(a)
        return result

