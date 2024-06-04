
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.list_gpu_processes)
class TorchCudaListGpuProcessesTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.9.0")
    def test_list_gpu_processes_0(self):
        
        a = 0
        result = torch.cuda.list_gpu_processes(a)
        return result
    @test_api_version.larger_than("1.9.0")
    def test_list_gpu_processes_1(self):
        
        a = 0
        result = torch.cuda.list_gpu_processes(device=a)
        return result

