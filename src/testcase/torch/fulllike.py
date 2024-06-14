import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.fulllike)
class TorchFulllikeTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_full_like_correctness(self):
    # Randomly generate the dimension and size of the input tensor
    dim = random.randint(1, 4)
    num_of_elements_each_dim = random.randint(1, 5)
    input_size = [num_of_elements_each_dim for i in range(dim)]

    # Generate a random input tensor
    input_tensor = torch.randn(input_size)
    # Generate a random fill value
    fill_value = random.uniform(-10.0, 10.0)

    result = torch.full_like(input_tensor, fill_value)
    return result
