
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.initial_seed)
class TorchCudaInitialSeedTestCase(TorBencherTestCaseBase):
    def test_initial_seed(self, input=None):
        if input is not None:
            result = torch.cuda.initial_seed()
            return [result, input]
        result = torch.cuda.initial_seed()
        return [result, None]

