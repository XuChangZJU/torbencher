import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.gather)
class TorchTensorGatherTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_gather_correctness(self):
        # Randomly generate tensor dimension
        dim = random.randint(1, 4)
        # Randomly generate number of elements for each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Generate input size
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Generate a random tensor with the specified input size
        input_tensor = torch.randn(input_size)
        # Generate a random dimension to gather from
        gather_dim = random.randint(0, dim - 1)
        # Generate index tensor size
        index_size = input_size.copy()
        # Modify the size of the gather dimension
        index_size[gather_dim] = random.randint(1, 5)
        # Generate random index tensor
        index_tensor = torch.randint(0, input_size[gather_dim], index_size)
        # Gather values from the input tensor
        result = input_tensor.gather(gather_dim, index_tensor)
        return result
    
    
    
    