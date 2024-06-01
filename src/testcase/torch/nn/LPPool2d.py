import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.LPPool2d)
class TorchNNLPPool2dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_lp_pool2d(self, input=None):
        if input is not None:
            result = torch.nn.LPPool2d(input[0], input[1])(input[2])
            return [result, input]
        a = torch.randn(1, 2, 4, 4)
        pool = torch.nn.LPPool2d(2, 3)
        result = pool(a)
        return [result, [2, 3, a]]

