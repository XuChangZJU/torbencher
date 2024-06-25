import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.index_reduce)
class TorchIndexreduceTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_index_reduce_correctness(self):
        dim = random.randint(0, 1)  # Random dimension for the reduction, limited to ensure valid indexing
        num_of_elements_each_dim = random.randint(2, 5)  # Random number of elements in each dimension
        input_size = [num_of_elements_each_dim for _ in range(2)]  # Generate random input size for 2D tensor
    
        input_tensor = torch.randn(input_size)  # Random input tensor
        index_size = input_size[dim]  # Ensure index tensor size matches the dimension being reduced
        index = torch.randint(0, input_size[dim], (input_size[1 - dim],))  # Random index tensor within bounds of input tensor along dimension 'dim'
        source = torch.randn(index.size())  # Random source tensor matching the size of index tensor
        reduce = random.choice(['prod', 'mean', 'amax', 'amin'])  # Random reduction operation
    
        result = torch.index_reduce(input_tensor, dim, index, source, reduce)
        return result
    
    
    
    