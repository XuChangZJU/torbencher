import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.cummin)
class TorchTensorCumminTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_cummin_correctness(self):
    dim = random.randint(0, 3)  # Random dimension for the cummin operation
    num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
    input_size = [num_of_elements_each_dim for _ in range(4)]  # Generate a 4D tensor with random sizes

    tensor = torch.randn(input_size)  # Generate a random tensor with the specified size
    result, indices = tensor.cummin(dim)  # Perform the cummin operation along the specified dimension
    return result, indices
