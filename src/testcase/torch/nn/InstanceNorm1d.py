
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.InstanceNorm1d)
class TorchNNInstanceNorm1dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_instance_norm1d(self, input=None):
        if input is not None:
            result = torch.nn.InstanceNorm1d(input[0])(input[1])
            return [result, input]
        a = torch.randn(10, 20)
        bn = torch.nn.InstanceNorm1d(20)
        result = bn(a)
        return [result, [20, a]]

