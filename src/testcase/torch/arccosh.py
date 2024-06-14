import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.arccosh)
class TorchArccoshTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_arccosh_correctness(self):
        # Generate a random dimension for the tensor
        dim = random.randint(1, 4)  
        # Generate a random number of elements for each dimension
        num_of_elements_each_dim = random.randint(1,5) 
        # Create a list representing the size of the tensor
        input_size=[num_of_elements_each_dim for i in range(dim)] 
        # Generate a random tensor with values greater than 1
        input_tensor = torch.rand(input_size) + 1
        result = torch.arccosh(input_tensor)
        return result
    