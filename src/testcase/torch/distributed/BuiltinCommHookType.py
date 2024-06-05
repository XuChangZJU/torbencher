
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.distributed.BuiltinCommHookType)
class TorchBuiltinCommHookTypeTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_builtin_comm_hook_type_correctness(self):
        result = torch.distributed.BuiltinCommHookType.pybind11_type()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_builtin_comm_hook_type_large_scale(self):
        result = torch.distributed.BuiltinCommHookType.pybind11_type()
        return result

