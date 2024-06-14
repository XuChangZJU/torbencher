import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.pad)
class TorchNnFunctionalPadTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_pad_correctness(self):
    # Random dimension for the tensors
    dim = random.randint(2, 5)
    # Random number of elements each dimension
    num_of_elements_each_dim = random.randint(1, 5)
    input_size = [num_of_elements_each_dim for i in range(dim)]
    # Generate random tensor with the specified size
    input_tensor = torch.randn(input_size)
    # Generate random padding size, making sure m/2 <= input dimensions and m is even
    m = 2 * random.randint(1, dim)
    pad = [random.randint(0, 3) for _ in range(m)]
    # Call pad function
    result = torch.nn.functional.pad(input_tensor, pad)
    return result
