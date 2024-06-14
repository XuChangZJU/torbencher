import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.foreachsin)
class TorchForeachsinTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_foreach_sin_correctness(self):
        num_of_tensors = random.randint(1, 5)  # Random number of tensors in the list
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements in each dimension
        
        input_size = [num_of_elements_each_dim for _ in range(dim)]
        
        tensor_list = [torch.randn(input_size) for _ in range(num_of_tensors)]
        torch._foreach_sin_(tensor_list)  # Apply in-place sine transformation
        
        return tensor_list
    