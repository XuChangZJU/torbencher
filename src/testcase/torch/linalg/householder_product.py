
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.linalg.householder_product)
class TorchLinalgHouseholderProductTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.10")
    def test_householder_product(self, input=None):
        if input is not None:
            result = torch.linalg.householder_product(input[0], input[1])
            return [result, input]
        a = torch.randn(3, 3)
        tau = torch.randn(3)
        result = torch.linalg.householder_product(a, tau)
        return [result, [a, tau]]

