import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.dist)
class TorchTensorDistTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_dist_correctness(self):
        # Random dimension for the tensors
        dim = random.randint(1, 4)  
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1,5) 
        # Random input size
        input_size=[num_of_elements_each_dim for i in range(dim)] 
        # Generate random tensor1
        tensor1 = torch.randn(input_size)
        # Generate random tensor2 with the same size as tensor1
        tensor2 = torch.randn(input_size)
        # Calculate the distance between tensor1 and tensor2
        result = tensor1.dist(tensor2)
        return result
    
    
    
    