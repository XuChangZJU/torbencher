import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.symite)
class TorchSymiteTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_sym_ite_correctness(self):
        # Generate random condition tensor with bool values (0 or 1)
        condition_dim = random.randint(1, 4)  # Random dimension for the condition tensor
        num_elements_each_dim_cond = random.randint(1, 5)  # Random number of elements for each dimension
        condition_size = [num_elements_each_dim_cond for _ in range(condition_dim)]
        condition = torch.randint(0, 2, condition_size).bool()
    
        # Generate random x tensor with the same size as condition tensor
        x = torch.randn(condition_size)
        
        # Generate random y tensor with the same size as condition tensor
        y = torch.randn(condition_size)
    
        # Apply torch.sym_ite to select between x and y tensor elements based on condition
        result = torch.sym_ite(condition, x, y)
        
        return result
    