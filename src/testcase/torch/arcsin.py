import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.arcsin)
class TorchArcsinTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_arcsin_correctness(self):
    dim = random.randint(1, 4)  
    num_of_elements_each_dim = random.randint(1,5) 
    input_size=[num_of_elements_each_dim for i in range(dim)] 

    input_tensor = torch.randn(input_size) # generate random tensor data
    input_tensor = input_tensor.clamp(min=-1., max=1.) # clamp the input tensor to [-1, 1] to make arcsin valid
    result = torch.arcsin(input_tensor)
    return result
