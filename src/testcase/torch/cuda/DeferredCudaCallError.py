
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.DeferredCudaCallError)
class TorchCudaDeferredCudaCallErrorTestCase(TorBencherTestCaseBase):
    def test_deferredcudacallerror_correctness(self):
        error = torch.cuda.DeferredCudaCallError(random.randint(0, 1000))
        result = error.type
        return result

    def test_deferredcudacallerror_large_scale(self):
        error = torch.cuda.DeferredCudaCallError(random.randint(0, 1000))
        result = error.type
        return result

