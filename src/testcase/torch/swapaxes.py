import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.swapaxes)
class TorchSwapaxesTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_swapaxes_correctness(self):
        dim = random.randint(2, 4) # dim should be larger than 2
        num_of_elements_each_dim = random.randint(1,5)
        input_size=[num_of_elements_each_dim for i in range(dim)] 
    
        input_tensor = torch.randn(input_size)
        axis0 = random.randint(0, dim - 1)
        axis1 = random.randint(0, dim - 1)
        result = torch.swapaxes(input_tensor, axis0, axis1)
        return result
    