import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.tril_indices)
class TorchTrilindicesTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_tril_indices_correctness(self):
        row = random.randint(1, 10)  # Random number of rows
        col = random.randint(1, 10)  # Random number of columns
        offset = random.randint(-min(row, col) + 1, max(row, col) - 1)  # Random offset within valid range
        result = torch.tril_indices(row, col, offset)
        return result
    
    
    
    
    
    
    