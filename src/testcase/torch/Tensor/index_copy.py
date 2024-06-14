import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.index_copy)
class TorchTensorIndexcopyTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_index_copy_correctness(self):
        dim = random.randint(0, 3)  # Random dimension to perform index_copy
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for _ in range(4)]  # Generate a 4D tensor
    
        tensor1 = torch.randn(input_size)  # Original tensor
        tensor2 = torch.randn(input_size)  # Tensor to copy from
    
        index_size = random.randint(1, num_of_elements_each_dim)  # Random size for index tensor
        index = torch.randint(0, num_of_elements_each_dim, (index_size,))  # Random index tensor
    
        result = tensor1.index_copy(dim, index, tensor2)
        return result
    
    
    
    