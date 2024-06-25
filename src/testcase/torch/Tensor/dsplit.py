import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.dsplit)
class TorchTensorDsplitTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_dsplit_correctness(self):
        dim = 3  # dsplit requires at least 3 dimensions
        num_of_elements_each_dim = random.randint(2, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for _ in range(dim)]
    
        tensor = torch.randn(input_size)
        
        # Ensure the split size is a divisor of the size of the dimension to be split
        split_size_or_sections = random.choice([i for i in range(1, num_of_elements_each_dim + 1) if num_of_elements_each_dim % i == 0])
    
        result = tensor.dsplit(split_size_or_sections)
        return result
    
    
    
    