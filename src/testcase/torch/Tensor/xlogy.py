import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.xlogy)
class TorchTensorXlogyTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_xlogy_correctness(self):
        """
        Test the correctness of xlogy with small scale random parameters.
        """
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
    
        # Generate random tensors
        tensor1 = torch.randn(input_size)  
        tensor2 = torch.randn(input_size)  
    
        # Calculate xlogy
        result = tensor1.xlogy(tensor2)
        return result
    
    
    
    