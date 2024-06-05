
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.PyTorchFileWriter)
class TorchPyTorchFileWriterTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_pytorchfilewriter_correctness(self):
        result = torch.PyTorchFileWriter()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_pytorchfilewriter_large_scale(self):
        result = torch.PyTorchFileWriter()
        return result

