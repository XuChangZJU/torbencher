import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.autograd.profiler_util.Kernel)
class TorchAutogradProfilerUutilKernelTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_Kernel_correctness(self):
        # Kernel(name, device, duration)
        name = "test_name"  # string, the name of the kernel
        device = "cuda" if torch.cuda.is_available() else "cpu"  # string, the device of the kernel
        duration = random.uniform(0.1, 10.0)  # float, the duration of the kernel

        kernel = torch.autograd.profiler_util.Kernel(name, device, duration)
        return kernel
