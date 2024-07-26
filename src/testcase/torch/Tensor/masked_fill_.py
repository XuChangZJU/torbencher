import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.masked_fill_)
class TorchTensorMaskedfillTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_masked_fill__correctness(self):
        # Randomly generate tensor dimension
        dim = random.randint(1, 4)
        # Randomly generate number of elements for each dimension
        num_of_elements_each_dim = random.randint(1,5)
        # Generate input size
        input_size=[num_of_elements_each_dim for i in range(dim)] 
    
        # Generate input tensor
        input_tensor = torch.randn(input_size)
        # Generate mask tensor with the same size as input tensor
        mask_tensor = torch.rand(input_size) > 0.5 # Generate random bool tensor
        # Generate value to fill
        value = random.uniform(0.1, 10.0)
        # Apply masked_fill_
        result = input_tensor.masked_fill_(mask_tensor, value)
        return result
    
    
    
    