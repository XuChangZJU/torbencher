
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.Upsample)
class TorchUpsampleTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_upsample_correctness(self):
        size = (random.randint(1, 10), random.randint(1, 10))
        scale_factor = random.uniform(0.1, 10.0)
        mode = random.choice(['nearest', 'linear', 'bilinear', 'trilinear'])
        align_corners = random.choice([True, False])
        input_tensor = torch.randn(random.randint(1, 10), random.randint(1, 10), random.randint(1, 10), random.randint(1, 10))
        upsample = torch.nn.Upsample(size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners)
        result = upsample(input_tensor)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_upsample_large_scale(self):
        size = (random.randint(100, 1000), random.randint(100, 1000))
        scale_factor = random.uniform(0.1, 10.0)
        mode = random.choice(['nearest', 'linear', 'bilinear', 'trilinear'])
        align_corners = random.choice([True, False])
        input_tensor = torch.randn(random.randint(1000, 10000), random.randint(100, 1000), random.randint(100, 1000), random.randint(100, 1000))
        upsample = torch.nn.Upsample(size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners)
        result = upsample(input_tensor)
        return result

