import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.backends.cudnn.is_available)
class TorchBackendsCudnnIsavailableTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_is_available_correctness(self):
        # No specific input parameters needed for torch.backends.cudnn.is_available
        result_cudnn_available = torch.backends.cudnn.is_available() # Check if CUDNN is available
        return result_cudnn_available
    