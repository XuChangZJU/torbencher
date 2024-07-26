import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.backends.cudnn.version)
class TorchBackendsCudnnVersionTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_backends_cudnn_version_correctness(self):
        # No input parameters needed for torch.backends.cudnn.version
        cudnn_version = torch.backends.cudnn.version()
        return cudnn_version
