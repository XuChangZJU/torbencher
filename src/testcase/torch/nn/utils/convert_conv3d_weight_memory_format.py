import unittest

import torch
import torch.nn as nn

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.utils.convert_conv3d_weight_memory_format)
class TorchNnUtilsConvertUconv3dUweightUmemoryUformatTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is not available")
    def test_convert_conv3d_weight_memory_format_correctness(self):
        """
        https://github.com/pytorch/pytorch/pull/131742
        """
        input = torch.randint(1, 10, (2, 8, 4, 4, 4), dtype=torch.float16, device="cuda")
        model = nn.Sequential(
            nn.Conv3d(8, 4, 3).cuda().half())
        model = nn.utils.convert_conv3d_weight_memory_format(model, torch.channels_last_3d)
        out = model(input)
        return out
