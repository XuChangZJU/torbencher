
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.special.xlog1py)
class TorchSpecialXlog1pyTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_xlog1py_0d(self, input=None):
        if input is not None:
            result = torch.special.xlog1py(input[0], input[1])
            return [result, input]
        a = torch.randn([])
        b = torch.randn([])
        result = torch.special.xlog1py(a, b)
        return [result, [a, b]]

    @test_api_version.larger_than("1.1.3")
    def test_xlog1py_1d(self, input=None):
        if input is not None:
            result = torch.special.xlog1py(input[0], input[1])
            return [result, input]
        a = torch.randn(5)
        b = torch.randn(5)
        result = torch.special.xlog1py(a, b)
        return [result, [a, b]]

