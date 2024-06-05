
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.ExternalStream)
class TorchCudaExternalStreamTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.9.0")
    def test_externalstream_correctness(self):
        stream = torch.cuda.ExternalStream()
        result = stream.priority_range
        return result

    @test_api_version.larger_than("1.9.0")
    def test_externalstream_large_scale(self):
        stream = torch.cuda.ExternalStream()
        result = stream.priority_range
        return result

