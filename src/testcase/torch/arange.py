import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.arange)
class TorchArangeTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_arange_correctness(self):
        # Randomly generate parameters for torch.arange
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
    
        start = random.uniform(-10.0, 10.0)  # Random start value between -10.0 and 10.0
        end = start + random.uniform(0.1, 10.0)  # Ensure end is greater than start
        step = random.uniform(0.1, 5.0)  # Random step value between 0.1 and 5.0
    
        # Call torch.arange with the generated parameters
        result = torch.arange(start, end, step)
        return result
    
    
    
    
    
    
    