import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.nantonum)
class TorchTensorNantonumTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_nan_to_num_correctness(self):
    # Randomly generate tensor dimension and size
    dim = random.randint(1, 4)
    num_of_elements_each_dim = random.randint(1, 5)
    input_size = [num_of_elements_each_dim for i in range(dim)]

    # Create a random tensor with NaN, positive infinity, and negative infinity values
    tensor = torch.randn(input_size)
    tensor[tensor > 0.5] = float('inf')  # Replace some values with positive infinity
    tensor[tensor < -0.5] = float('-inf')  # Replace some values with negative infinity
    tensor[tensor.abs() < 0.1] = float('nan')  # Replace some values with NaN

    # Call nan_to_num to replace NaN, positive infinity, and negative infinity values
    result = tensor.nan_to_num()
    return result
