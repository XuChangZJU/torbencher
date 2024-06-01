import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.log)
class TorchLogTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_log_4d(self, input=None):
        if input is not None:
            result = torch.log(input[0])
            return [result, input]
        a = torch.randn(4).abs() + 1
        result = torch.log(a)
        return [result, [a]]

