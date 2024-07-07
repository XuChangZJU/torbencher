import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cpu.amp.autocast)
class TorchCpuAmpAutocastTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_autocast_correctness(self):
        # Randomly generate input tensor size
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
    
        # Generate random input tensors
        tensor1 = torch.randn(input_size)
        tensor2 = torch.randn(input_size)
    
        # Perform operation within autocast context
        with torch.cpu.amp.autocast():
            result = tensor1 + tensor2
    
        return result
    
    
    
    