import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.logcumsumexp)
class TorchTensorLogcumsumexpTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_logcumsumexp_correctness(self):
        # Create random dimension for the tensor
        dim = random.randint(1, 4)
        # Create random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Create input size
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Generate a random tensor with the specified input size
        input_tensor = torch.randn(input_size)
        # Randomly select a dimension for cummulative sum
        dim = random.randint(0, len(input_size) - 1)
        # Calculate the logcumsumexp of the tensor along the specified dimension
        result = input_tensor.logcumsumexp(dim)
        return result
    