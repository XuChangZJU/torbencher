import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.isnan)
class TorchIsnanTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_isnan_correctness(self):
    # Generate random dimension and size for the tensor
    dim = random.randint(1, 4)
    num_of_elements_each_dim = random.randint(1, 5)
    input_size = [num_of_elements_each_dim for i in range(dim)]

    # Create a tensor with some NaN values
    input_tensor = torch.randn(input_size)
    # Randomly select some elements to be NaN
    mask = torch.randint(0, 2, input_size, dtype=torch.bool) 
    input_tensor[mask] = float('nan')

    result = torch.isnan(input_tensor)
    return result
