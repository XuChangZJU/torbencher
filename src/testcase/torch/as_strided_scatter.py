
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.as_strided_scatter)
class TorchAsStridedScatterTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_as_strided_scatter_correctness(self):
        dim = random.randint(1, 10)
        tensor = torch.randn(dim)
        src = torch.randn(dim)
        size = random.randint(1, 10)
        stride = random.randint(1, 10)
        result = torch.as_strided_scatter(tensor, src, size=(size,), stride=(stride,))
        return result

    @test_api_version.larger_than("1.1.3")
    def test_as_strided_scatter_large_scale(self):
        dim = random.randint(1000, 10000)
        tensor = torch.randn(dim)
        src = torch.randn(dim)
        size = random.randint(1, 10)
        stride = random.randint(1, 10)
        result = torch.as_strided_scatter(tensor, src, size=(size,), stride=(stride,))
        return result

