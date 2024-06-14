import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.add)
class TorchTensorAddTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_torch_Tensor_add_correctness(self):
        # Generate random dimension and size for the tensors
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
    
        # Generate random tensors
        self_tensor = torch.randn(input_size)  
        other_tensor = torch.randn(input_size) # other_tensor with the same shape as self_tensor to ensure broadcasting compatibility
    
        # Call the add method
        result = self_tensor.add(other_tensor)
        return result
    
    
    
    