import random
import unittest
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.quantized_max_pool1d)
class TorchQuantizedUmaxUpool1dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_quantized_max_pool1d_correctness(self):
        qx = torch.quantize_per_tensor(torch.rand(2, 2), 1.5, 3, torch.quint8)
        return torch.quantized_max_pool1d(qx, [2])