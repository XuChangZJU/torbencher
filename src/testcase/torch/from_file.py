import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.from_file)
class TorchFromUfileTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_from_file_correctness(self):
        t = torch.randn(2, 5, dtype=torch.float64)
        t.to("cpu").numpy().tofile('storage.bin')
        t_mapped = torch.from_file('storage.pt', shared=False, size=10, dtype=torch.float64)
        return t_mapped
