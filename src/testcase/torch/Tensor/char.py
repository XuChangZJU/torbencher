import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.char)
class TorchTensorCharTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_char_correctness(self):
        # Generate random dimension and number of elements for the tensor
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Create a random tensor 
        input_tensor = torch.randn(input_size)

        # Call the char() function on the tensor
        result = input_tensor.char()

        return result
