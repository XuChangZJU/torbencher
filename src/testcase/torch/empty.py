import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.empty)
class TorchEmptyTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_empty_correctness(self):
        # Randomly generate the size of the tensor
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
    
        # Generate an empty tensor
        result = torch.empty(input_size)
        return result
    
    
    
    
    
    
    