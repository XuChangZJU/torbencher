
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.set_rng_state_all)
class TorchCudaSetRngStateAllTestCase(TorBencherTestCaseBase):
    def test_set_rng_state_all_correctness(self):
        rng_state_all = torch.cuda.get_rng_state_all()
        result = torch.cuda.set_rng_state_all(rng_state_all)
        return result

    def test_set_rng_state_all_large_scale(self):
        rng_state_all = torch.cuda.get_rng_state_all()
        result = torch.cuda.set_rng_state_all(rng_state_all)
        return result

