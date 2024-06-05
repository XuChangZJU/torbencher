
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.ZeroPad3d)
class TorchZeroPad3dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_zeropad3d_correctness(self):
        padding = (random.randint(1, 10), random.randint(1, 10), random.randint(1, 10), random.randint(1, 10), random.randint(1, 10), random.randint(1, 10))
        input_tensor = torch.randn(random.randint(1, 10), random.randint(1, 10), random.randint(1, 10), random.randint(1, 10), random.randint(1, 10), random.randint(1, 10))
        zero_pad = torch.nn.ZeroPad3d(padding)
        result = zero_pad(input_tensor)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_zeropad3d_large_scale(self):
        padding = (random.randint(100, 1000), random.randint(100, 1000), random.randint(100, 1000), random.randint(100, 1000), random.randint(100, 1000), random.randint(100, 1000))
        input_tensor = torch.randn(random.randint(1000, 10000), random.randint(100, 1000), random.randint(100, 1000), random.randint(100, 1000), random.randint(100, 1000), random.randint(100, 1000))
        zero_pad = torch.nn.ZeroPad3d(padding)
        result = zero_pad(input_tensor)
        return result



import torch
import random

