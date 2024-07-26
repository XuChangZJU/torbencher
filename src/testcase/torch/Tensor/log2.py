import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.log2)
class TorchTensorLog2TestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_log2_correctness(self):
        """
        Test the correctness of torch.Tensor.log2.
        """
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
        input_tensor = torch.randn(input_size)  # Generate random tensor
        input_tensor = torch.abs(
            input_tensor)  # Make sure all elements are positive to avoid log2 of zero or negative values
        result = input_tensor.log2()
        return result
