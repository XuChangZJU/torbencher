
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.chain_matmul)
class TorchChain_matmulTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_chain_matmul(self, input=None):
        if input is not None:
            result = torch.chain_matmul(*input[0])
            return [result, input]
        a = torch.randn(3, 2, 2)
        b = torch.randn(2, 5, 6)
        c = torch.randn(6, 2, 2)
        result = torch.chain_matmul(a, b, c)
        return [result, [[a, b, c]]]

