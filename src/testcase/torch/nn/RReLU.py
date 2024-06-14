import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.RReLU)
class TorchNnRreluTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_rrelu_correctness(self):
    # Random dimension for the tensors
    dim = random.randint(1, 4)  
    # Random number of elements each dimension
    num_of_elements_each_dim = random.randint(1,5) 
    input_size=[num_of_elements_each_dim for i in range(dim)] 

    # Generate random input tensor
    input_tensor = torch.randn(input_size)
    # Generate random lower and upper bound
    lower = random.uniform(0, 0.5)
    upper = random.uniform(lower, 1) # upper bound should be larger than lower bound
    
    # Define and apply RReLU module
    rrelu_module = torch.nn.RReLU(lower, upper)
    output_tensor = rrelu_module(input_tensor)

    return output_tensor
