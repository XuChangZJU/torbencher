import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cdist)
class TorchCdistTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_cdist_correctness(self):
    batch_size = random.randint(1, 10)
    m = random.randint(1, 10)
    p = random.randint(1, 10)
    r = random.randint(1, 10)
    x1 = torch.randn(batch_size, p, m)
    x2 = torch.randn(batch_size, r, m)
    result = torch.cdist(x1, x2)
    return result
