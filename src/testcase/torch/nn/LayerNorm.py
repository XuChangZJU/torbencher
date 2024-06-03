
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.LayerNorm)
class TorchNNLayerNormTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_layer_norm(self, input=None):
        if input is not None:
            result = torch.nn.LayerNorm(input[0])(input[1])
            return result
        a = torch.randn(20, 5, 10, 10)
        norm = torch.nn.LayerNorm([5, 10, 10])
        result = norm(a)
        return result

