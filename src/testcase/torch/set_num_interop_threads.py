import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.set_num_interop_threads)
class TorchSetnuminteropthreadsTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_set_num_interop_threads(self):
        # Randomly selected number of interop threads
        num_threads = random.randint(1, 16)  # Range chosen based on typical thread counts for small and medium systems
        
        # Ensure no parallel work has started before setting interop threads
        torch.set_num_interop_threads(num_threads)
        
        # Create a tensor operation to observe interop thread effect
        tensor_size = [random.randint(1, 5) for _ in range(random.randint(1, 4))]  # Random tensor dimensions and sizes
        tensor = torch.randn(tensor_size)
    
        return torch.square(tensor)  # Simple operation reflecting interop threading setting
    
    
    
    