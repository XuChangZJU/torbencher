import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.isclose)
class TorchTensorIscloseTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_isclose_correctness(self):
        # Random dimension for the tensors
        dim = random.randint(1, 4)  
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1,5) 
        # Generate random input size
        input_size=[num_of_elements_each_dim for i in range(dim)] 
    
        # Generate random tensors
        tensor1 = torch.randn(input_size)
        tensor2 = tensor1 * (1 + random.uniform(-1e-05, 1e-05)) # Ensure tensor2 is close to tensor1
    
        # Calculate isclose result
        result = tensor1.isclose(tensor2)
        return result
    