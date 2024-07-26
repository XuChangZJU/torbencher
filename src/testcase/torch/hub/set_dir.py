import torch
import random
import os

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.hub.set_dir)
class TorchHubSetdirTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_set_dir_correctness(self):
        # torch.hub.set_dir(d: str)
        d = './test_hub'  # path to a local folder to save downloaded models & weights.
        result = torch.hub.set_dir(d)
        return result
