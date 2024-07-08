import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.backends.cuda.is_built)
class TorchBackendsCudaIsbuiltTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_is_built_correctness(self):
        # No input parameters needed for torch.backends.cuda.is_built
        result = torch.backends.cuda.is_built()
        return result
    