
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.cast)
class TorchCudaCastTestCase(TorBencherTestCaseBase):
    def test_cast_correctness(self):
        dim = random.randint(1, 10)
        tensor = torch.randn(dim, dtype=torch.float)
        result = torch.cuda.cast(tensor, dtype=torch.double)
        return result

    def test_cast_large_scale(self):
        dim = random.randint(1000, 10000)
        tensor = torch.randn(dim, dtype=torch.float)
        result = torch.cuda.cast(tensor, dtype=torch.double)
        return result

