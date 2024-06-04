
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.get_rng_state)
class TorchCudaGetRngStateTestCase(TorBencherTestCaseBase):
    def test_get_rng_state(self):
        result = torch.cuda.get_rng_state()
        return result

