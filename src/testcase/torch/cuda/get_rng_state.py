
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.get_rng_state)
class TorchCudaGetRngStateTestCase(TorBencherTestCaseBase):
    def test_get_rng_state_correctness(self):
        device = random.randint(0, torch.cuda.device_count() - 1)
        result = torch.cuda.get_rng_state(device)
        return result

    def test_get_rng_state_large_scale(self):
        device = random.randint(0, torch.cuda.device_count() - 1)
        result = torch.cuda.get_rng_state(device)
        return result

