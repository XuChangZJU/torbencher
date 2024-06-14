import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.set)
class TorchTensorSetTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_set_correctness(self):
        # Random dimension for the tensors
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]
    
        # Create a random tensor as the source
        source_tensor = torch.randn(input_size)
        
        # Create a target tensor with different size
        target_tensor = torch.empty(0)
        
        # Apply set_ to make target_tensor share the same storage, size, and strides as source_tensor
        result_tensor = target_tensor.set_(source_tensor)
        
        return result_tensor
    