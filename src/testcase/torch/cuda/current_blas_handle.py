
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.current_blas_handle)
class TorchCudaCurrentBlasHandleTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.8.0")
    def test_current_blas_handle_correctness(self):
        result = torch.cuda.current_blas_handle()
        return result

    @test_api_version.larger_than("1.8.0")
    def test_current_blas_handle_large_scale(self):
        result = torch.cuda.current_blas_handle()
        return result

