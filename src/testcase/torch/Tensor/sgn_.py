import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.sgn_)
class TorchTensorSgnTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_sgn__correctness(self):
        """
        Test the correctness of the torch.Tensor.sgn_() operator.
        """
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1,5) # Random number of elements each dimension
        input_size=[num_of_elements_each_dim for i in range(dim)] 
    
        input_tensor = torch.randn(input_size) # Create a random tensor
        input_tensor_copy = torch.clone(input_tensor) # Create a copy for comparison
        input_tensor.sgn_() # Apply sgn_() operation in-place
        return input_tensor # Return the tensor after applying sgn_()
    
    
    
    