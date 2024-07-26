import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.masked_select)
class TorchTensorMaskedselectTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_masked_select_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1,5) # Random number of elements each dimension
        input_size=[num_of_elements_each_dim for i in range(dim)] 
    
        input_tensor = torch.randn(input_size)
        # Generate a random boolean mask with the same shape as the input tensor
        mask = torch.randint(0, 2, size=input_size, dtype=torch.bool) 
        result = input_tensor.masked_select(mask)
        return result
    
    
    
    