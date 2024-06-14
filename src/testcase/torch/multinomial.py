import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.multinomial)
class TorchMultinomialTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_multinomial_correctness(self):
        rows = random.randint(1, 4)  # Number of rows for the input tensor
        cols = random.randint(2, 10)  # Minimum number of samples to draw from the distribution
        input_tensor = torch.abs(torch.randn(rows, cols))  # Ensure non-negative values for probabilities
        num_samples = random.randint(1, cols - 1)  # Ensure it's possible to draw unique samples without replacement
        
        result = torch.multinomial(input_tensor, num_samples)
        return result
    
    
    
    