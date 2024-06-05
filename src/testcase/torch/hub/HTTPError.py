
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.hub.HTTPError)
class TorchHubHTTPErrorTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_getcode_correctness(self):
        error = torch.hub.HTTPError(url="https://example.com", code=404)  # Replace with a valid URL
        result = error.getcode()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_getcode_large_scale(self):
        error = torch.hub.HTTPError(url="https://example.com", code=500)  # Replace with a valid URL
        result = error.getcode()
        return result

