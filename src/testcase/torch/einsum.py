
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.einsum)
class TorchEinsumTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_einsum_4d(self, input=None):
        if input is not None:
            result = torch.einsum(input[0], input[1])
            return [result, input]
        a = torch.randn(4, 4)
        b = torch.randn(4, 4)
        result = torch.einsum('ij,jk->ik', [a, b])
        return [result, ['ij,jk->ik', [a, b]]]

