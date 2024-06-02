
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.InstanceNorm2d)
class TorchNNInstanceNorm2dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_instance_norm2d(self, input=None):
        if input is not None:
            result = torch.nn.InstanceNorm2d(input[0])(input[1])
            return [result, input]
        a = torch.randn(10, 20, 30, 40)
        bn = torch.nn.InstanceNorm2d(20)
        result = bn(a)
        return [result, [20, a]]

