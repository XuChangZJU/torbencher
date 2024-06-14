import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.bincount)
class TorchBincountTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_bincount_correctness(self):
        # Generate random non-negative integer tensor, ensuring it is one-dimensional
        num_elements = random.randint(1, 10)  # Random number of elements in the tensor
        input_tensor = torch.randint(0, 10, (num_elements,), dtype=torch.int32)
    
        # Generate optional weights tensor of the same size as input_tensor
        if random.choice([True, False]):
            weights = torch.randn(num_elements)
            result = torch.bincount(input_tensor, weights)
        else:
            result = torch.bincount(input_tensor)
    
        return result
    
    
    
    