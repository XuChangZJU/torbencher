
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.distributed.FileStore)
class TorchFileStoreTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_file_store_correctness(self):
        file_path = 'test_file_store.txt'
        result = torch.distributed.FileStore.pybind11_type(file_path)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_file_store_large_scale(self):
        file_path = 'test_file_store.txt'
        result = torch.distributed.FileStore.pybind11_type(file_path)
        return result

