
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.MaxUnpool3d)
class TorchNNMaxUnpool3dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_max_unpool3d(self, input=None):
        if input is not None:
            result = torch.nn.MaxUnpool3d(input[0])(input[1])
            return [result, input]
        a = torch.randn(1, 1, 2, 2, 2)
        indices = torch.tensor([[[[[0, 1], [1, 2]]]]], dtype=torch.long)
        pool = torch.nn.MaxUnpool3d(2)
        result = pool(a, indices)
        return [result, [2, a, indices]]

