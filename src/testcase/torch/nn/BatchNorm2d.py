import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.BatchNorm2d)
class TorchNNBatchNorm2dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_batch_norm2d(self, input=None):
        if input is not None:
            result = torch.nn.BatchNorm2d(input[0])(input[1])
            return [result, input]
        a = torch.randn(10, 100, 10, 10)
        bn = torch.nn.BatchNorm2d(100)
        result = bn(a)
        return [result, [100, a]]
