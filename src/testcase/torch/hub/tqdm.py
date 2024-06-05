
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.hub.tqdm)
class TorchHubTqdmTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_update_correctness(self):
        progress_bar = torch.hub.tqdm(total=100)
        progress_bar.update(10)
        result = progress_bar.n
        return result

    @test_api_version.larger_than("1.1.3")
    def test_update_large_scale(self):
        progress_bar = torch.hub.tqdm(total=10000)
        progress_bar.update(1000)
        result = progress_bar.n
        return result

