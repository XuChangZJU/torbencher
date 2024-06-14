import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.xlogy)
class TorchXlogyTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_xlogy_correctness(self):
        # Define the dimension and size of the input tensors
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
    
        # Generate random tensors
        input_tensor = torch.randn(input_size)  
        other_tensor = torch.randn(input_size)  
    
        # Calculate xlogy
        result = torch.xlogy(input_tensor, other_tensor)
        return result
    