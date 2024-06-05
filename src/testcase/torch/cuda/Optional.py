
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.Optional)
class TorchCudaOptionalTestCase(TorBencherTestCaseBase):
    def test_optional_correctness(self):
        result = torch.cuda.Optional
        return result

    def test_optional_large_scale(self):
        result = torch.cuda.Optional
        return result

