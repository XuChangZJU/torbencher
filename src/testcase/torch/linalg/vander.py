
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.linalg.vander)
class TorchLinalgVanderTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_vander_0(self, input=None):
        if input is not None:
            result = torch.linalg.vander(input[0])
            return [result, input]
        x = torch.tensor([1, 2, 3, 5])
        result = torch.linalg.vander(x)
        return [result, [x]]

    @test_api_version.larger_than("2.0.0")
    def test_vander_1(self, input=None):
        if input is not None:
            result = torch.linalg.vander(input[0], N=input[1])
            return [result, input]
        x = torch.tensor([1, 2, 3, 5])
        result = torch.linalg.vander(x, N=3)
        return [result, [x, 3]]

