import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.get_num_interop_threads)
class TorchGetUnumUinteropUthreadsTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_get_num_interop_threads_correctness(self):
        # No input parameters to randomize for torch.get_num_interop_threads
        num_threads = torch.get_num_interop_threads()
        return num_threads
