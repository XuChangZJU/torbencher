import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.foreachcosh)
class TorchForeachcoshTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_foreach_cosh_correctness(self):
    # Generate random input parameters for torch._foreach_cosh_
    dim = random.randint(1, 4)  # Random dimension for the tensors
    num_of_elements_each_dim = random.randint(1,5) # Random number of elements each dimension
    input_size=[num_of_elements_each_dim for i in range(dim)] 
    list_len = random.randint(1, 5) # Random length of the input list
    input_tensors = [torch.randn(input_size) for _ in range(list_len)]

    # Apply torch._foreach_cosh_
    torch._foreach_cosh_(input_tensors)

    # Return one of the tensors to check the effect
    return input_tensors[0]
