import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cpu.StreamContext)
class TorchCpuStreamcontextTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_stream_context_correctness(self):
        # Randomly select a stream priority
        priority = random.randint(-10, 0)  # Random priority between -10 and 0 (valid range for priorities)
        stream = torch.cuda.Stream(device='cuda', priority=priority)
        
        # Create a tensor and perform an operation within the stream context
        tensor_size = [random.randint(1, 5) for _ in range(random.randint(1, 4))]  # Random tensor size
        tensor = torch.randn(tensor_size, device='cuda')
        
        with torch.cuda.stream(stream):
            result = tensor * 2  # Example operation within the stream context
        
        return result
    
    
    
    