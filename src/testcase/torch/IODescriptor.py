
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.IODescriptor)
class TorchIODescriptorTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_iodescriptor_correctness(self):
        result = torch.IODescriptor()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_iodescriptor_large_scale(self):
        result = torch.IODescriptor()
        return result

