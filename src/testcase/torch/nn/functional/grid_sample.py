
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.grid_sample)
class TorchNNFunctionalGridSampleTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_grid_sample_common(self, input=None):
        if input is not None:
            result = torch.nn.functional.grid_sample(input[0], input[1], mode=input[2], padding_mode=input[3])
            return [result, input]
        a = torch.randn(1, 1, 3, 3)
        b = torch.tensor([[[[0.0000, 1.0000], [0.0000, 0.0000]], [[0.0000, 0.0000], [0.0000, 1.0000]]]])
        c = 'bilinear'
        d = 'zeros'
        result = torch.nn.functional.grid_sample(a, b, mode=c, padding_mode=d)
        return [result, [a, b, c, d]]


