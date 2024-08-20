import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.fake_quantize_per_tensor_affine)
class TorchFakeUquantizeUperUtensorUaffineTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_fake_quantize_per_tensor_affine_correctness(self):
        x = torch.randn(4)
        return x, torch.fake_quantize_per_tensor_affine(x, 0.1, 0, 0, 255), torch.fake_quantize_per_tensor_affine(x, torch.tensor(0.1), torch.tensor(0), 0, 255)