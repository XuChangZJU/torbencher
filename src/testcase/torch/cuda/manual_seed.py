
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.manual_seed)
class TorchCudaManualSeedTestCase(TorBencherTestCaseBase):
    def test_manual_seed_correctness(self):
        seed = random.randint(0, 10000)
        device = random.randint(0, torch.cuda.device_count() - 1)
        result = torch.cuda.manual_seed(seed, device)
        return result

    def test_manual_seed_large_scale(self):
        seed = random.randint(0, 10000)
        device = random.randint(0, torch.cuda.device_count() - 1)
        result = torch.cuda.manual_seed(seed, device)
        return result

