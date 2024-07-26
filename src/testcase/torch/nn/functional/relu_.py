import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.relu_)
class TorchNnFunctionalReluTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_relu_correctness(self):
        # Randomly generate the dimension and size of the input tensor
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
    
        # Generate a random tensor with values ranging from -10.0 to 10.0
        input_tensor = torch.randn(input_size) * 10
    
        # Apply the relu_ operation in-place
        torch.nn.functional.relu_(input_tensor)
        
        return input_tensor
    
    
    
    