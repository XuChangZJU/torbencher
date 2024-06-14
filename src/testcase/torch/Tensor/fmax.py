import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.fmax)
class TorchTensorFmaxTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_fmax_correctness(self):
        # Generate random dimension and size for the tensors
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
    
        # Generate random tensors
        tensor1 = torch.randn(input_size)
        tensor2 = torch.randn(input_size)
        
        # Calculate element-wise maximum
        result = tensor1.fmax(tensor2)
        
        return result
    