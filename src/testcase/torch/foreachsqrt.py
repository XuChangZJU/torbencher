import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.foreachsqrt)
class TorchForeachsqrtTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_foreach_sqrt_correctness(self):
    # foreach_sqrt requires the input to be a list of tensors
    num_of_tensors = random.randint(1, 5)  
    dim = random.randint(1, 4)
    num_of_elements_each_dim = random.randint(1, 5)
    input_size = [num_of_elements_each_dim for i in range(dim)]

    # Generate a list of random tensors
    tensor_list = [torch.randn(input_size) for _ in range(num_of_tensors)]
    result = torch._foreach_sqrt(tensor_list)
    return result
