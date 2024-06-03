
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.set_rng_state)
class TorchCudaSetRngStateTestCase(TorBencherTestCaseBase):
    def test_set_rng_state(self, input=None):
        if input is not None:
            result = torch.cuda.set_rng_state(input[0])
            return result
        a = torch.cuda.get_rng_state()
        result = torch.cuda.set_rng_state(a)
        return result

