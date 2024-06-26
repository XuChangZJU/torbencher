
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.max_unpool3d)
class TorchNNFunctionalMaxUnpool3dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_max_unpool3d_common(self, input=None):
        if input is not None:
            result = torch.nn.functional.max_unpool3d(input[0], input[1], kernel_size=input[2], stride=input[3], padding=input[4], output_size=input[5])
            return [result, input]
        a = torch.tensor([[[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]]], dtype=torch.float)
        b = torch.tensor([[[[[1, 3], [2, 4]]]]], dtype=torch.long)
        c = 2
        d = 2
        e = 0
        f = (1, 1, 5, 5, 5)
        result = torch.nn.functional.max_unpool3d(a, b, kernel_size=c, stride=d, padding=e, output_size=f)
        return [result, [a, b, c, d, e, f]]


