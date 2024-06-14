import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.less)
class TorchLessTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_less_correctness(self):
        # Generate random dimensions for the tensors
        dim = random.randint(1, 4)
        # Generate random number of elements for each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Create input_size list for tensor shape
        input_size = [num_of_elements_each_dim for i in range(dim)]
    
        # Generate random tensors of the same shape
        input1 = torch.randn(input_size)
        input2 = torch.randn(input_size)
        # Apply torch.less
        result = torch.less(input1, input2)
        return result
    
    
    
    
    
    
    