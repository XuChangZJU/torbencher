import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.fmax)
class TorchFmaxTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_fmax_correctness(self):
        # Random dimensions for the tensors
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]
    
        # Generate two random tensors
        input_tensor = torch.randn(input_size)
        other_tensor = torch.randn(input_size)
    
        # Introduce NaN values randomly in both tensors to test NaN handling
        num_nans = random.randint(0, num_of_elements_each_dim)
        for _ in range(num_nans):
            idx = tuple([random.randint(0, num_of_elements_each_dim-1) for _ in range(dim)])
            if random.choice([True, False]):
                input_tensor[idx] = float('nan')
            else:
                other_tensor[idx] = float('nan')
    
        # Perform fmax operation
        result = torch.fmax(input_tensor, other_tensor)
        
        return result
    
    
    
    