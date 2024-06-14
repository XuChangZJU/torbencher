import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.argwhere)
class TorchArgwhereTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_argwhere_correctness(self):
        # Randomly generate the dimension of the tensor
        dim = random.randint(1, 4)
        # Randomly generate the number of elements in each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Create a list of input sizes for the tensor
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Generate a random tensor with values between -10 and 10
        input_tensor = torch.randint(-10, 10, input_size) 
        # Calculate the indices of non-zero elements using torch.argwhere
        result = torch.argwhere(input_tensor)
        return result
    
    
    
    
    
    
    