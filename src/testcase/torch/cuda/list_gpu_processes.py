
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.list_gpu_processes)
class TorchCudaListGpuProcessesTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.10.0")
    def test_list_gpu_processes_correctness(self):
        result = torch.cuda.list_gpu_processes()
        return result

    @test_api_version.larger_than("1.10.0")
    def test_list_gpu_processes_large_scale(self):
        result = torch.cuda.list_gpu_processes()
        return result

