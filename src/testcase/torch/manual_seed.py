
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.manual_seed)
class TorchManualSeedTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_manual_seed_correctness(self):
        seed = random.randint(0, 1000)
        result = torch.manual_seed(seed)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_manual_seed_large_scale(self):
        seed = random.randint(0, 1000)
        result = torch.manual_seed(seed)
        return result

