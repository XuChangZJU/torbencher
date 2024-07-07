import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.compiler_deepdive)
class TorchCompilerdeepdiveTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_compiler_deepdive_correctness(self):
        # Randomly generate dimensions for the tensor
        dim = random.randint(1, 4)
        # Randomly generate the number of elements in each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]
    
        # Generate a random tensor with the specified dimensions
        tensor = torch.randn(input_size)
        
        # Assuming the function to be tested is torch.compile
        result = torch.compile(tensor)
        return result
    
    
    
    