
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.matmul)
class TorchMatmulTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_matmul_1d_1d(self):
        a = torch.randn(3)
        b = torch.randn(3)
        result = torch.matmul(a, b)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_matmul_2d_2d(self):
        a = torch.randn(3, 4)
        b = torch.randn(4, 5)
        result = torch.matmul(a, b)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_matmul_2d_1d(self):
        a = torch.randn(3, 4)
        b = torch.randn(4)
        result = torch.matmul(a, b)
        return result

