
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.synchronize)
class TorchCudaSynchronizeTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_synchronize_0(self, input=None):
        if input is not None:
            result = torch.cuda.synchronize()
            return [result, input]
        result = torch.cuda.synchronize()
        return [result, None]

    @test_api_version.larger_than("1.1.3")
    def test_synchronize_1(self, input=None):
        if input is not None:
            result = torch.cuda.synchronize(input[0])
            return [result, input]
        a = 0
        result = torch.cuda.synchronize(a)
        return [result, [a]]

    @test_api_version.larger_than("1.1.3")
    def test_synchronize_2(self, input=None):
        if input is not None:
            result = torch.cuda.synchronize(device=input[0])
            return [result, input]
        a = 0
        result = torch.cuda.synchronize(device=a)
        return [result, [a]]

