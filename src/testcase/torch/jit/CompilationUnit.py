
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.jit.CompilationUnit)
class TorchJitCompilationUnitTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_compilation_unit_correctness(self):
        comp_unit = torch.jit.CompilationUnit()
        result = comp_unit.pybind11_type
        return result

    @test_api_version.larger_than("1.1.3")
    def test_compilation_unit_large_scale(self):
        comp_unit = torch.jit.CompilationUnit()
        result = comp_unit.pybind11_type
        return result

