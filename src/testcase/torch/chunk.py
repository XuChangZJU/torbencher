
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.chunk)
class TorchChunkTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_chunk_correctness(self):
        tensor = torch.randn(random.randint(1, 10), random.randint(1, 10))
        chunks = random.randint(1, 10)
        result = torch.chunk(tensor, chunks)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_chunk_large_scale(self):
        tensor = torch.randn(random.randint(1000, 10000), random.randint(1000, 10000))
        chunks = random.randint(1000, 10000)
        result = torch.chunk(tensor, chunks)
        return result

