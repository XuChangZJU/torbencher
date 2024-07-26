import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.LocalResponseNorm)
class TorchNnLocalresponsenormTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_LocalResponseNorm_correctness(self):
        # Random input size
        num_of_dimensions = random.randint(3, 6)  
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(num_of_dimensions)]
    
        # Random N, C
        N = random.randint(1, 10)
        C = random.randint(1, 10)
        input_size[0] = N
        input_size[1] = C
    
        # Random size
        size = random.randint(1, C) # size should be less than or equal to number of channels C
    
        input_tensor = torch.randn(input_size)
        lrn = torch.nn.LocalResponseNorm(size)
        output_tensor = lrn(input_tensor)
        return output_tensor
    
    
    
    