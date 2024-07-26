import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.arcsinh)
class TorchArcsinhTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_arcsinh_correctness(self):
        """
        Check the correctness of torch.arcsinh.
        """
        dim = random.randint(1, 4)  
        num_of_elements_each_dim = random.randint(1,5) 
        input_size=[num_of_elements_each_dim for i in range(dim)] 
    
        input_tensor = torch.randn(input_size) 
        result = torch.arcsinh(input_tensor)
        return result
    
    
    
    
    
    
    