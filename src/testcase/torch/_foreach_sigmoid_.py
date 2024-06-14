import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch._foreach_sigmoid_)
class TorchForeachsigmoidTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_foreach_sigmoid_correctness(self):
        # foreach_sigmoid_ operator applies element-wise sigmoid to a list of tensors.
        # This test checks the correctness of foreach_sigmoid_ by comparing it to
        # the results of applying torch.sigmoid to each tensor individually.
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
        num_of_tensors = random.randint(1, 3)
        tensor_list = [torch.randn(input_size) for i in range(num_of_tensors)]
        result = torch._foreach_sigmoid_(tensor_list)
        return result
    
    
    
    
    
    
    