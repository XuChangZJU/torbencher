import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.scatteradd)
class TorchTensorScatteraddTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_scatter_add_correctness(self):
        # Define the dimension of the tensor
        dim = random.randint(0, 3)  
        # Define the size of the tensor
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim + 1)] 
    
        # Generate random input tensor 
        input = torch.randn(input_size)
        # Generate random index tensor with values in range [0, input_size[dim])
        index = torch.randint(0, input_size[dim], input_size)
        # Generate random source tensor with the same size as input
        src = torch.randn(input_size)
        result = input.scatter_add(dim, index, src)
        return result
    