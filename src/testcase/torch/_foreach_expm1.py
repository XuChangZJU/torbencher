import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch._foreach_expm1)
class TorchForeachexpm1TestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_foreach_expm1_correctness(self):
        # foreach_expm1 requires the length of input list to be larger than 0
        input_list_length = random.randint(1, 5)
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
        input_tensors = [torch.randn(input_size) for _ in range(input_list_length)]
        result = torch._foreach_expm1(input_tensors)
        return result
    
    
    
    
    
    
    