import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.expm1)
class TorchExpm1TestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_expm1_correctness(self):
    # Generate random dimension and size for the input tensor
    dim = random.randint(1, 4)
    num_of_elements_each_dim = random.randint(1, 5)
    input_size = [num_of_elements_each_dim for i in range(dim)]

    # Generate a random tensor 
    input_tensor = torch.randn(input_size)

    # Calculate expm1
    result = torch.expm1(input_tensor)
    return result
