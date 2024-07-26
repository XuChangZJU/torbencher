import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.dropout2d)
class TorchNnFunctionalDropout2dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_dropout2d_correctness(self):
        dim = random.randint(2, 4)  # Random dimension for the tensors, at least 2 for 2D
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]
    
        # Generate random input tensor
        input_tensor = torch.randn(input_size)
    
        # Generate random p value
        p = random.uniform(0.0, 1.0)  # Random p value between 0.0 and 1.0
    
        # Apply dropout2d
        result = torch.nn.functional.dropout2d(input_tensor, p)
        return result
    
    
    
    