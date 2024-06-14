import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.utils.weight_norm)
class TorchNnUtilsWeightnormTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_weight_norm_correctness(self):
        # Define the dimensions of the weight tensor
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
    
        # Create a Linear layer for testing
        m = torch.nn.Linear(in_features=random.randint(1, 10), out_features=input_size[0])
    
        # Apply weight normalization to the Linear layer
        wn = torch.nn.utils.weight_norm(m)
    
        # Return the weight normalized module
        return wn
        
    
    
    
    