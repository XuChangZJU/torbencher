
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.from_file)
class TorchFromFileTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_from_file_correctness(self):
        filename = "test.bin"
        data = torch.randn(random.randint(1, 10))
        torch.save(data, filename)
        result = torch.from_file(filename)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_from_file_large_scale(self):
        filename = "test.bin"
        data = torch.randn(random.randint(1000, 10000))
        torch.save(data, filename)
        result = torch.from_file(filename)
        return result

