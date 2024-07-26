import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.squeeze)
class TorchSqueezeTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_squeeze_correctness(self):
        # Generate random dimension and size for the input tensor
        dim = random.randint(2, 5)  
        # Generate random size for each dimension, ensuring at least one dimension has size 1
        input_size = [random.randint(1, 5) if i != random.randint(0, dim - 1) else 1 for i in range(dim)]  
    
        input_tensor = torch.randn(input_size)
        result = torch.squeeze(input_tensor)
        return result
    
    
    
    
    
    
    