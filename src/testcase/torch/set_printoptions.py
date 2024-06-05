
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.set_printoptions)
class TorchSetPringOptionsTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_set_printoptions_correctness(self):
        precision = random.randint(1, 10)
        threshold = random.randint(1, 10)
        edgeitems = random.randint(1, 10)
        linewidth = random.randint(1, 10)
        result = torch.set_printoptions(precision=precision, threshold=threshold, edgeitems=edgeitems, linewidth=linewidth)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_set_printoptions_large_scale(self):
        precision = random.randint(1, 10)
        threshold = random.randint(1, 10)
        edgeitems = random.randint(1, 10)
        linewidth = random.randint(1, 10)
        result = torch.set_printoptions(precision=precision, threshold=threshold, edgeitems=edgeitems, linewidth=linewidth)
        return result

