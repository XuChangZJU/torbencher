
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.fake_quantize_per_channel_affine)
class TorchFakeQuantizePerChannelAffineTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_fake_quantize_per_channel_affine(self, input=None):
        if input is not None:
            result = torch.fake_quantize_per_channel_affine(
                input[0], input[1], input[2], input[3], input[4]
            )
            return [result, input]
        a = torch.randn(2, 3, 4, 4)
        b = torch.randn(3)
        c = torch.randint(-3, 3, size=(3,))
        d = 0
        e = 255
        result = torch.fake_quantize_per_channel_affine(a, b, c, d, e)
        return [result, [a, b, c, d, e]]


