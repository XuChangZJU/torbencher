import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.indexput)
class TorchTensorIndexputTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_index_put_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensor
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for _ in range(dim)]
    
        tensor = torch.randn(input_size)  # Random tensor
        indices = tuple(torch.randint(0, num_of_elements_each_dim, (random.randint(1, num_of_elements_each_dim),)) for _ in range(dim))  # Random indices
        values = torch.randn(indices[0].size())  # Values tensor with the same size as the first index tensor
    
        result = tensor.index_put_(indices, values)  # Perform the index_put_ operation
        return result
    