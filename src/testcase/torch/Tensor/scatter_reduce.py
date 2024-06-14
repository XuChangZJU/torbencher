import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.scatter_reduce)
class TorchTensorScatterreduceTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_scatter_reduce_correctness(self):
        # Define the dimension of the tensor
        dim = random.randint(0, 3) # Dimension to reduce along
        # Define the size of the tensor
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(4)]
        # Create random tensors
        input = torch.randn(input_size)
        src = torch.randn(input_size)
        # Generate random indices
        index = torch.randint(0, input_size[dim], input_size)
        # Define the reduction operation
        reduce = random.choice(["sum", "prod", "mean", "min", "max", "amax", "amin"])
        # Perform scatter_reduce operation
        result = torch.scatter_reduce(input, dim, index, src, reduce)
        return result
    
    
    
    