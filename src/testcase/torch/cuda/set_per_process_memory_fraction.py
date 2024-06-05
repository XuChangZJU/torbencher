
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.set_per_process_memory_fraction)
class TorchCudaSetPerProcessMemoryFractionTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.5.0")
    def test_set_per_process_memory_fraction_correctness(self):
        fraction = random.uniform(0.0, 1.0)
        device = random.randint(0, torch.cuda.device_count() - 1)
        result = torch.cuda.set_per_process_memory_fraction(fraction, device)
        return result

    @test_api_version.larger_than("1.5.0")
    def test_set_per_process_memory_fraction_large_scale(self):
        fraction = random.uniform(0.0, 1.0)
        device = random.randint(0, torch.cuda.device_count() - 1)
        result = torch.cuda.set_per_process_memory_fraction(fraction, device)
        return result

