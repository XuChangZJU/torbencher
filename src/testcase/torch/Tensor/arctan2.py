import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.arctan2)
class TorchTensorArctan2TestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_arctan2_correctness(self):
        """
        Test the correctness of torch.Tensor.arctan2.
        """
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1,5) # Random number of elements each dimension
        input_size=[num_of_elements_each_dim for i in range(dim)] 
    
        input_tensor = torch.randn(input_size)  # Random tensor
        other_tensor = torch.randn(input_size)  # Random tensor
        result = input_tensor.arctan2(other_tensor)
        return result
    
    
    
    