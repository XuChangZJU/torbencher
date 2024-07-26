import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.logical_or)
class TorchTensorLogicalorTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_logical_or_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1,5) # Random number of elements each dimension
        input_size=[num_of_elements_each_dim for i in range(dim)] 
    
        input1 = torch.randn(input_size) > 0 # generate random tensor with element True or False
        input2 = torch.randn(input_size) > 0 # generate random tensor with element True or False
        result = input1.logical_or(input2)
        return result
    
    
    
    