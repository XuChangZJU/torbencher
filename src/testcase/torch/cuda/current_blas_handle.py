
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.current_blas_handle)
class TorchCudaCurrentBlasHandleTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_current_blas_handle(self, input=None):
        if input is not None:
            result = torch.cuda.current_blas_handle()
            return [result, input]
        result = torch.cuda.current_blas_handle()
        return [result, None]


