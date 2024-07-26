import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.backends.cuda.cufft_plan_cache.torch.backends.cuda.preferred_linalg_library)
class TorchBackendsCudaCufftplancacheTorchBackendsCudaPreferredlinalglibraryTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_preferred_linalg_library_correctness(self):
        # No input parameters for torch.backends.cuda.preferred_linalg_library
        result = torch.backends.cuda.preferred_linalg_library()
        return result
