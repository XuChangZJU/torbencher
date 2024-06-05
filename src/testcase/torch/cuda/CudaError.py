
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.CudaError)
class TorchCudaCudaErrorTestCase(TorBencherTestCaseBase):
    def test_cudaerror_correctness(self):
        error = torch.cuda.CudaError(random.randint(0, 1000))
        result = error.type
        return result

    def test_cudaerror_large_scale(self):
        error = torch.cuda.CudaError(random.randint(0, 1000))
        result = error.type
        return result

