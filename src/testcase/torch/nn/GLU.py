
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.GLU)
class TorchNNGLUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_glu(self):
        
        a = torch.randn(1, 2, 4)
        glu = torch.nn.GLU()
        result = glu(a)
        return result

