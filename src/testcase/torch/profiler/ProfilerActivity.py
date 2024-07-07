import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.profiler.ProfilerActivity)
class TorchProfilerProfileractivityTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_profiler_activity_cpu(self):
        # Test case for CPU activity
        activity = torch.profiler.ProfilerActivity.CPU
        tensor_size = [random.randint(1, 5) for _ in range(random.randint(1, 4))]
        tensor = torch.randn(tensor_size)
        result = tensor * 2  # Simple operation to show CPU activity
        return result
    
    def test_profiler_activity_cuda(self):
        # Test case for CUDA activity
        if torch.cuda.is_available():
            activity = torch.profiler.ProfilerActivity.CUDA
            tensor_size = [random.randint(1, 5) for _ in range(random.randint(1, 4))]
            tensor = torch.randn(tensor_size, device='cuda')
            result = tensor * 2  # Simple operation to show CUDA activity
            return result
    
    def test_profiler_activity_xpu(self):
        # Test case for XPU activity (assuming XPU is available and supported)
        # Note: PyTorch does not currently support XPU, this is a hypothetical example
        activity = torch.profiler.ProfilerActivity.XPU
        tensor_size = [random.randint(1, 5) for _ in range(random.randint(1, 4))]
        tensor = torch.randn(tensor_size)
        result = tensor * 2  # Simple operation to show XPU activity
        return result
    
    def test_profiler_activity_mtia(self):
        # Test case for MTIA activity (assuming MTIA is available and supported)
        # Note: PyTorch does not currently support MTIA, this is a hypothetical example
        activity = torch.profiler.ProfilerActivity.MTIA
        tensor_size = [random.randint(1, 5) for _ in range(random.randint(1, 4))]
        tensor = torch.randn(tensor_size)
        result = tensor * 2  # Simple operation to show MTIA activity
        return result
    
    
    
    
    
    
    