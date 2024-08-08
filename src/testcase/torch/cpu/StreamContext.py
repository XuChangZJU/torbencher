import torch
import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version


class TorchCpuStreamcontextTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_stream_context_correctness(self):

        stream = torch.cpu.Stream()

        tensor_size = [random.randint(1, 5) for _ in range(random.randint(1, 4))]
        tensor = torch.randn(tensor_size)

        with torch.cpu.StreamContext(stream):
            result = tensor * 2

        return result
