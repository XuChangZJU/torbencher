import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.jit.enable_onednn_fusion)
class TorchJitEnableUonednnUfusionTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_enable_onednn_fusion_correctness(self):
        # Randomly enable or disable onednn JIT fusion
        enabled_status = random.choice([True, False])
        result = torch.jit.enable_onednn_fusion(enabled_status)
        return result
