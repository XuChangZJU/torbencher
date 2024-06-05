
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.is_bf16_supported)
class TorchCudaIsBf16SupportedTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.10.0")
    def test_is_bf16_supported_correctness(self):
        device = random.randint(0, torch.cuda.device_count() - 1)
        result = torch.cuda.is_bf16_supported(device)
        return result

    @test_api_version.larger_than("1.10.0")
    def test_is_bf16_supported_large_scale(self):
        device = random.randint(0, torch.cuda.device_count() - 1)
        result = torch.cuda.is_bf16_supported(device)
        return result

