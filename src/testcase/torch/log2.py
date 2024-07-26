import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.log2)
class TorchLog2TestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_log2_correctness(self):
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        input_tensor = torch.rand(input_size)  # generate random tensor with positive values for valid log2 calculation
        result = torch.log2(input_tensor)
        return result
