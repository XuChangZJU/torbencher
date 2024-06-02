
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.get_rng_state_all)
class TorchCudaGetRngStateAllTestCase(TorBencherTestCaseBase):
    def test_get_rng_state_all(self, input=None):
        if input is not None:
            result = torch.cuda.get_rng_state_all()
            return [result, input]
        result = torch.cuda.get_rng_state_all()
        return [result, None]





