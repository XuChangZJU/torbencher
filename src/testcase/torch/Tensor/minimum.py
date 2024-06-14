import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.minimum)
class TorchTensorMinimumTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_minimum_correctness(self):
        """
        Test the correctness of torch.Tensor.minimum with small scale random parameters.
        """
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
    
        tensor1 = torch.randn(input_size)  # Random tensor 1
        tensor2 = torch.randn(input_size)  # Random tensor 2
        result = tensor1.minimum(tensor2)  # Calculate element-wise minimum
        return result
    
    
    
    