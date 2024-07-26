import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.uniform_)
class TorchTensorUniformTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_uniform__correctness(self):
        # Random dimension for the tensors
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Random input size
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Generate a random tensor
        input_tensor = torch.randn(input_size)
        # Random from value
        from_value = random.uniform(-10.0, 10.0)
        # Random to value, make sure to > from
        to_value = random.uniform(from_value, from_value + 10.0)
        # Call the function
        result = input_tensor.uniform_(from_value, to_value)
        return result
