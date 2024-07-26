import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.fliplr)
class TorchTensorFliplrTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_fliplr_correctness(self):
        # Random dimension for the tensors (at least 2 dimensions are required for fliplr)
        dim = random.randint(2, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Generate input size
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Generate random tensor 
        input_tensor = torch.randn(input_size)
        # Flip the input tensor along the last dimension
        result = input_tensor.fliplr()
        return result
