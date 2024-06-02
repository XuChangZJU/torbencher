
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.fake_quantize_per_tensor_affine)
class TorchFakeQuantizePerTensorAffineTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_fake_quantize_per_tensor_affine(self, input=None):
        if input is not None:
            result = torch.fake_quantize_per_tensor_affine(
                input[0], input[1], input[2], input[3], input[4]
            )
            return [result, input]
        a = torch.randn(2, 3, 4, 4)
        b = 0.1
        c = 1
        d = 0
        e = 255
        result = torch.fake_quantize_per_tensor_affine(a, b, c, d, e)
        return [result, [a, b, c, d, e]]


