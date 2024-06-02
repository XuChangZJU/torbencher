
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.chunk)
class TorchChunkTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_chunk_4d(self, input=None):
        if input is not None:
            result = torch.chunk(input[0], input[1])
            return [result, input]
        a = torch.randn(8)
        result = torch.chunk(a, 4)
        return [result, [a, 4]]

