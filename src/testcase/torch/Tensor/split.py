import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.split)
class TorchTensorSplitTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_split_correctness(self):
        # Random dimension for the tensor
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Random input size
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Generate a random tensor
        tensor = torch.randn(input_size)
        # Random split_size, should be a divisor of the corresponding dimension size
        split_size = random.randint(1, num_of_elements_each_dim)
        # Random dim
        dim = random.randint(0, len(input_size) - 1)
        # Split the tensor
        result = torch.Tensor.split(tensor, split_size, dim)
        return result
    