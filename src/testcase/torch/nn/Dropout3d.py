
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.Dropout3d)
class TorchDropout3dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_dropout3d_correctness(self):
        p = random.uniform(0.0, 1.0)
        input_tensor = torch.randn(random.randint(1, 10), random.randint(1, 10), random.randint(1, 10), random.randint(1, 10), random.randint(1, 10))
        dropout = torch.nn.Dropout3d(p=p)
        result = dropout(input_tensor)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_dropout3d_large_scale(self):
        p = random.uniform(0.0, 1.0)
        input_tensor = torch.randn(random.randint(1000, 10000), random.randint(100, 1000), random.randint(100, 1000), random.randint(100, 1000), random.randint(100, 1000))
        dropout = torch.nn.Dropout3d(p=p)
        result = dropout(input_tensor)
        return result

