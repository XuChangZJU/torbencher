
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.as_strided_copy)
class TorchAsStridedCopyTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_as_strided_copy_correctness(self):
        dim = random.randint(1, 10)
        tensor = torch.randn(dim)
        size = random.randint(1, 10)
        stride = random.randint(1, 10)
        result = torch.as_strided_copy(tensor, size=(size,), stride=(stride,))
        return result

    @test_api_version.larger_than("1.1.3")
    def test_as_strided_copy_large_scale(self):
        dim = random.randint(1000, 10000)
        tensor = torch.randn(dim)
        size = random.randint(1, 10)
        stride = random.randint(1, 10)
        result = torch.as_strided_copy(tensor, size=(size,), stride=(stride,))
        return result

