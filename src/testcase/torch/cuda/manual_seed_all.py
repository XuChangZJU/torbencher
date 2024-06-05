
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.manual_seed_all)
class TorchCudaManualSeedAllTestCase(TorBencherTestCaseBase):
    def test_manual_seed_all_correctness(self):
        seed = random.randint(0, 10000)
        result = torch.cuda.manual_seed_all(seed)
        return result

    def test_manual_seed_all_large_scale(self):
        seed = random.randint(0, 10000)
        result = torch.cuda.manual_seed_all(seed)
        return result

