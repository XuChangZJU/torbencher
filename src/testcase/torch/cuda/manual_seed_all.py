
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.manual_seed_all)
class TorchCudaManualSeedAllTestCase(TorBencherTestCaseBase):
    def test_manual_seed_all(self, input=None):
        if input is not None:
            result = torch.cuda.manual_seed_all(input[0])
            return [result, input]
        a = 10
        result = torch.cuda.manual_seed_all(a)
        return [result, [a]]
