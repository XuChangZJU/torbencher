import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cpu.stream)
class TorchCpuStreamTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_cpu_stream_correctness(self):
        # Create a random tensor
        dim = random.randint(1, 4)  # Random dimension for the tensor
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for _ in range(dim)]
        
        tensor = torch.randn(input_size)
        
        # Create a CPU stream
        stream = torch.cpu.Stream()
        
        # Perform an operation within the stream context
        with torch.cpu.stream(stream):
            result = tensor + 1  # Simple operation to show the effect of the stream
        
        return result
    
    
    
    