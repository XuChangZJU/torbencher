
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.glu)
class TorchNNFunctionalGLUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_glu_common(self, input=None):
        if input is not None:
            result = torch.nn.functional.glu(input[0], dim=input[1])
            return [result, input]
        a = torch.randn(4, 8)
        b = 1
        result = torch.nn.functional.glu(a, dim=b)
        return [result, [a, b]]


