import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.divide)
class TorchTensorDivideTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_torch_Tensor_divide_correctness(self):
        """
        Test the correctness of torch.Tensor.divide.
        """
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1,5) # Random number of elements each dimension
        input_size=[num_of_elements_each_dim for i in range(dim)] 
    
        # Generate random tensors with random dimensions
        tensor1 = torch.randn(input_size)
        # Make sure the divisor is not zero to avoid ZeroDivisionError
        tensor2 = torch.randn(input_size) + 1e-6 
        result = tensor1.divide(tensor2)
        return result
    
    
    
    