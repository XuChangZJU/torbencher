
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cosine_similarity)
class TorchCosine_similarityTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_cosine_similarity(self, input=None):
        if input is not None:
            result = torch.cosine_similarity(input[0], input[1], input[2])
            return [result, input]
        a = torch.randn(100, 128)
        b = torch.randn(100, 128)
        result = torch.cosine_similarity(a, b, dim=1, eps=1e-8)
        return [result, [a, b, 1]]

