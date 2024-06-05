
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.distributed.PrefixStore)
class TorchPrefixStoreTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_prefix_store_correctness(self):
        result = torch.distributed.PrefixStore.pybind11_type()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_prefix_store_large_scale(self):
        result = torch.distributed.PrefixStore.pybind11_type()
        return result

