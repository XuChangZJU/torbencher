import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.Softsign)
class TorchNnSoftsignTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_nn_Softsign_correctness(self):
        # Define the dimension and size of the tensor
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
    
        # Generate a random tensor
        input_tensor = torch.randn(input_size)
    
        # Define the Softsign module
        softsign_function = torch.nn.Softsign()
    
        # Apply the Softsign function to the input tensor
        output_tensor = softsign_function(input_tensor)
        
        # Return the output tensor
        return output_tensor
    
    
    
    