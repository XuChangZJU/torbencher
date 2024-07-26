import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.diag_embed)
class TorchDiagembedTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_diag_embed_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]
        input_tensor = torch.randn(input_size)
        offset = random.randint(-(num_of_elements_each_dim - 1), num_of_elements_each_dim - 1)  # Random offset within valid range
        result = torch.diag_embed(input_tensor, offset)
        return result
    
    
    
    
    
    
    