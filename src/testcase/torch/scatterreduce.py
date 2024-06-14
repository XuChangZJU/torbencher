import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.scatterreduce)
class TorchScatterreduceTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_scatter_reduce_correctness(self):
        # Define the dimension of the input tensor
        dim = random.randint(0, 3)  
        # Define the size of each dimension for the input tensor
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim + 1)]  
    
        # Generate random tensors and parameters
        input_tensor = torch.randn(input_size)
        index_tensor = torch.randint(0, input_size[dim], size=(input_size)) # The values in index should be in the range [0, input_size[dim]-1]
        src_tensor = torch.randn(input_size)
        reduce_op = random.choice(["sum", "prod", "min", "max", "mean"])  
    
        # Perform scatter_reduce operation
        result = torch.scatter_reduce(input_tensor, dim, index_tensor, src_tensor, reduce_op)
        return result
    