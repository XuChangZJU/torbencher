
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.hub.Request)
class TorchHubRequestTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_add_header_correctness(self):
        request = torch.hub.Request()
        header = {"User-Agent": "PyTorch Hub Test"}
        request.add_header(header["User-Agent"], "PyTorch Hub Test")
        result = request.get_header(header["User-Agent"])
        return result

    @test_api_version.larger_than("1.1.3")
    def test_add_header_large_scale(self):
        request = torch.hub.Request()
        headers = {"User-Agent": "PyTorch Hub Test", "Accept": "application/json"}
        for key, value in headers.items():
            request.add_header(key, value)
        result = request.header_items()
        return result

