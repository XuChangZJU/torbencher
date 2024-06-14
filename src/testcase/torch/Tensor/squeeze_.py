import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.squeeze_)
class TorchTensorSqueezeTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_squeeze__correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1,5) # Random number of elements each dimension
        input_size=[num_of_elements_each_dim for i in range(dim)] 
        input_size.insert(random.randint(0,len(input_size)), 1) # Randomly insert a dimension of size 1
        input_tensor = torch.randn(input_size)
        input_tensor_copy = input_tensor.clone()
        input_tensor_copy.squeeze_(random.randint(0,len(input_size)-1)) # Randomly select a dimension to squeeze
        return input_tensor_copy
    
    
    
    