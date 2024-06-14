import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.repeatinterleave)
class TorchRepeatinterleaveTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_repeat_interleave_correctness(self):
    # Random dimension for the tensor
    dim = random.randint(1, 4)
    # Random number of elements each dimension
    num_of_elements_each_dim = random.randint(1, 5)
    input_size = [num_of_elements_each_dim for _ in range(dim)]
    
    # Generating random tensor
    input_tensor = torch.randn(input_size)
    # Generating random repeats tensor or integer
    if random.choice([True, False]):
        repeats = random.randint(1, 3)
    else:
        repeats = torch.randint(1, 4, (num_of_elements_each_dim,))

    # Randomly select a dimension
    dim_option = random.choice([None, random.randint(0, dim - 1)])
    
    # Applying repeat_interleave
    if dim_option is None:
        result = torch.repeat_interleave(input_tensor, repeats)
    else:
        result = torch.repeat_interleave(input_tensor, repeats, dim=dim_option)
    
    return result
