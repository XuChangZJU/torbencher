import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.repeatinterleave)
class TorchTensorRepeatinterleaveTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_repeat_interleave_correctness(self):
    # Random dimension for the tensors
    dim = random.randint(1, 4)
    # Random number of elements each dimension
    num_of_elements_each_dim = random.randint(1, 5)
    # Random input size
    input_size = [num_of_elements_each_dim for i in range(dim)]
    # Randomly generated tensor
    input_tensor = torch.randn(input_size)
    # Randomly generated repeats (must be positive integer)
    repeats = random.randint(1, 5)
    # Randomly generated dim (must be integer between -dim and dim-1)
    dim = random.randint(-dim, dim - 1)
    result = input_tensor.repeat_interleave(repeats, dim)
    return result
