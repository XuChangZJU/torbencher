
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.local_response_norm)
class TorchNNFunctionalLocalResponseNormTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_local_response_norm_common(self, input=None):
        if input is not None:
            result = torch.nn.functional.local_response_norm(input[0], input[1], alpha=input[2], beta=input[3], k=input[4])
            return [result, input]
        a = torch.randn(1, 5, 20, 20)
        b = 3
        c = 1e-04
        d = 0.75
        e = 2
        result = torch.nn.functional.local_response_norm(a, b, alpha=c, beta=d, k=e)
        return [result, [a, b, c, d, e]]


