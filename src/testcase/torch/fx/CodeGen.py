
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.fx.CodeGen)
class TorchFxCodeGenTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.11")
    def test_codegen_correctness(self):
        # TODO: No concrete test case for CodeGen.type due to its abstract nature.
        return None

    @test_api_version.larger_than("1.11")
    def test_codegen_large_scale(self):
        # TODO: No concrete test case for CodeGen.type due to its abstract nature.
        return None

    @test_api_version.larger_than("1.11")
    def test_codegen_additional_globals_correctness(self):
        # TODO: No concrete test case for CodeGen.additional_globals due to its abstract nature.
        return None

    @test_api_version.larger_than("1.11")
    def test_codegen_additional_globals_large_scale(self):
        # TODO: No concrete test case for CodeGen.additional_globals due to its abstract nature.
        return None

    @test_api_version.larger_than("1.11")
    def test_codegen_gen_fn_def_correctness(self):
        # TODO: No concrete test case for CodeGen.gen_fn_def due to its abstract nature.
        return None

    @test_api_version.larger_than("1.11")
    def test_codegen_gen_fn_def_large_scale(self):
        # TODO: No concrete test case for CodeGen.gen_fn_def due to its abstract nature.
        return None

    @test_api_version.larger_than("1.11")
    def test_codegen_generate_output_correctness(self):
        # TODO: No concrete test case for CodeGen.generate_output due to its abstract nature.
        return None

    @test_api_version.larger_than("1.11")
    def test_codegen_generate_output_large_scale(self):
        # TODO: No concrete test case for CodeGen.generate_output due to its abstract nature.
        return None

    @test_api_version.larger_than("1.11")
    def test_codegen_process_inputs_correctness(self):
        # TODO: No concrete test case for CodeGen.process_inputs due to its abstract nature.
        return None

    @test_api_version.larger_than("1.11")
    def test_codegen_process_inputs_large_scale(self):
        # TODO: No concrete test case for CodeGen.process_inputs due to its abstract nature.
        return None

    @test_api_version.larger_than("1.11")
    def test_codegen_process_outputs_correctness(self):
        # TODO: No concrete test case for CodeGen.process_outputs due to its abstract nature.
        return None

    @test_api_version.larger_than("1.11")
    def test_codegen_process_outputs_large_scale(self):
        # TODO: No concrete test case for CodeGen.process_outputs due to its abstract nature.
        return None


