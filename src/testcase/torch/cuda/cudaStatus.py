
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.cudaStatus)
class TorchCudaCudaStatusTestCase(TorBencherTestCaseBase):
    def test_cudastatus_correctness(self):
        status = torch.cuda.cudaStatus(random.randint(0, 1000))
        result = status.type
        return result

    def test_cudastatus_large_scale(self):
        status = torch.cuda.cudaStatus(random.randint(0, 1000))
        result = status.type
        return result



