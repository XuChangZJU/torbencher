import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.utils.data.torch.utils.data.get_worker_info)
class TorchUtilsDataTorchUtilsDataGetworkerinfoTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_get_worker_info_correctness(self):
        # No parameters to randomize for this function.
        result = torch.utils.data.get_worker_info()
        return result
