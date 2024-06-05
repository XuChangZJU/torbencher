
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.mem_get_info)
class TorchCudaMemGetInfoTestCase(TorBencherTestCaseBase):
    def test_mem_get_info_correctness(self):
        device = random.randint(0, torch.cuda.device_count() - 1)
        result = torch.cuda.mem_get_info(device)
        return result

    def test_mem_get_info_large_scale(self):
        device = random.randint(0, torch.cuda.device_count() - 1)
        result = torch.cuda.mem_get_info(device)
        return result

