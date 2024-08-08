import os

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.hub.get_dir)
class TorchHubGetUdirTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_get_dir_correctness(self):
        # No random parameters needed for this test case
        torch.hub.set_dir(os.path.join(torch.hub._get_torch_home(), 'test_hub'))
        result = torch.hub.get_dir()
        return result
