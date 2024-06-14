import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.indexadd)
class TorchTensorIndexaddTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_index_add_correctness(self):
        dim = random.randint(0, 3)  # Random dimension for the operation
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for _ in range(4)]  # Fixed 4D tensor for simplicity
    
        tensor = torch.randn(input_size)  # Random input tensor
        source = torch.randn(input_size)  # Random source tensor
    
        # Generate a valid index tensor
        index_size = input_size[dim]  # Index size should match the dimension size of the input tensor
        index = torch.randint(0, input_size[dim], (index_size,))  # Random index tensor
    
        result = tensor.index_add(dim, index, source)
        return result
    