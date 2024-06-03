
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.linalg.vector_norm.vector_norm)
class TorchLinalgVectorNormVectorNormTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.11.0")
    def test_vector_norm(self, input=None):
        if input is not None:
            result = torch.linalg.vector_norm.vector_norm(input[0], ord=input[1])
            return result
        a = torch.randn(3)
        result = torch.linalg.vector_norm.vector_norm(a, ord=2)
        return result
