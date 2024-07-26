import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.new_ones)
class TorchTensorNewonesTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_new_ones_correctness(self):
        # Define the size of the tensor
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        size = [num_of_elements_each_dim for i in range(dim)]

        # Create a random tensor
        input_tensor = torch.randn(size)

        # Call new_ones on the tensor
        result = input_tensor.new_ones(size)
        return result
