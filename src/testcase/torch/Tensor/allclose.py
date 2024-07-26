import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.allclose)
class TorchTensorAllcloseTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_allclose_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1,5) # Random number of elements each dimension
        input_size=[num_of_elements_each_dim for i in range(dim)] 
    
        tensor1 = torch.randn(input_size)
        tensor2 = tensor1.clone() # Create tensor2 by cloning tensor1 to ensure they are initially close
        result = tensor1.allclose(tensor2)
        return result
    
    def test_allclose_not_close(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1,5) # Random number of elements each dimension
        input_size=[num_of_elements_each_dim for i in range(dim)] 
    
        tensor1 = torch.randn(input_size)
        tensor2 = torch.randn(input_size) # Generate a completely different tensor
        result = tensor1.allclose(tensor2)
        return result
    
    