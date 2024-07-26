import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.Softshrink)
class TorchNnSoftshrinkTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_softshrink_correctness(self):
        """
        Test the correctness of torch.nn.Softshrink by comparing the output with the expected result.
        """
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]

        input_tensor = torch.randn(input_size)
        softshrink = torch.nn.Softshrink()
        output_tensor = softshrink(input_tensor)
        return output_tensor
