import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.special.logit)
class TorchSpecialLogitTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_logit_correctness(self):
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1,5)
        input_size=[num_of_elements_each_dim for i in range(dim)]
    
        input_tensor = torch.rand(input_size) # generate random tensor with value between (0, 1)
        eps = random.uniform(1e-7, 1e-5) # generate random eps
        result = torch.special.logit(input_tensor, eps)
        return result
    
    
    
    