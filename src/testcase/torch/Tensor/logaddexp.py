import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.logaddexp)
class TorchTensorLogaddexpTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_logaddexp_correctness(self):
        # Randomly generate the dimension of the input tensors
        dim = random.randint(1, 4)
        # Randomly generate the number of elements in each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Create the input size list for the tensors
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate random tensors of the same size
        input_tensor = torch.randn(input_size)
        other_tensor = torch.randn(input_size)
        # Calculate the element-wise logaddexp
        result = input_tensor.logaddexp(other_tensor)
        return result
