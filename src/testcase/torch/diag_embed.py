
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.diag_embed)
class TorchDiagEmbedTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_diag_embed(self, input=None):
        if input is not None:
            result = torch.diag_embed(input[0], offset=input[1], dim1=input[2], dim2=input[3])
            return [result, input]
        a = torch.randn(2, 3)
        result = torch.diag_embed(a, offset=1, dim1=-2, dim2=-1)
        return [result, [a, 1, -2, -1]]

