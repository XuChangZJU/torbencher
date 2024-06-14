import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.gather)
class TorchGatherTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_gather_correctness(self):
        dim = random.randint(0, 2)  # Random dimension to index along
        input_size = [random.randint(1, 5) for _ in range(3)]  # Random size for the input tensor
        index_size = input_size.copy()
        index_size[dim] = random.randint(1, input_size[dim])  # Ensure index size is valid
        input_tensor = torch.randn(input_size)
        index_tensor = torch.randint(0, input_size[dim], size=index_size)
        result = torch.gather(input_tensor, dim, index_tensor)
        return result
    
    
    
    
    
    
    