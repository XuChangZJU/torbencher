
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.compiled_with_cxx11_abi)
class TorchCompiledWithCxx11AbiTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_compiled_with_cxx11_abi_correctness(self):
        result = torch.compiled_with_cxx11_abi()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_compiled_with_cxx11_abi_large_scale(self):
        result = torch.compiled_with_cxx11_abi()
        return result

