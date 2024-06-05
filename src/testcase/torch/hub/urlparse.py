
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.hub.urlparse)
class TorchHubUrlparseTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_urlparse_correctness(self):
        url = "https://example.com/path/to/file.txt"
        result = torch.hub.urlparse(url)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_urlparse_large_scale(self):
        url = "https://verylongdomainname.example.com/path/with/many/subdirectories/and/a/very/long/filename.txt"
        result = torch.hub.urlparse(url)
        return result

