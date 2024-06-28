import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.index_reduce)
class TorchTensorIndexreduceTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_index_reduce_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for _ in range(dim)]
    
        tensor = torch.randn(input_size)
        index_dim = random.randint(0, dim - 1)  # Random dimension to reduce along
        index_size = input_size[index_dim]
        index = torch.randint(0, index_size, (index_size,))  # Random indices within the range of the dimension size
        source = torch.randn(input_size)  # Source tensor with the same shape as the input tensor
    
        result = tensor.index_reduce(index_dim, index, source, reduce='prod')
        return result
    
    
    
    