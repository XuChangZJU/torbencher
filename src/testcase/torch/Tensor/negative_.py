import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.negative_)
class TorchTensorNegativeTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_negative__correctness(self):
        """
        Test the correctness of torch.Tensor.negative_() by comparing it with the output of torch.Tensor.negative().
        """
        dim = random.randint(1, 4)  
        num_of_elements_each_dim = random.randint(1, 5) 
        input_size = [num_of_elements_each_dim for i in range(dim)] 
    
        tensor = torch.randn(input_size)
        tensor_copy = tensor.clone()  # Create a copy to keep the original tensor for comparison
        tensor_copy.negative_() 
        result = tensor_copy
        return result
    
    
    
    