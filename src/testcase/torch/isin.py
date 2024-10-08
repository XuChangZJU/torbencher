import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.isin)
class TorchIsinTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_isin_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]

        elements = torch.randint(0, 10, input_size)  # elements tensor with values between 0 and 9
        test_elements = torch.randint(0, 10, input_size)  # test_elements tensor with values between 0 and 9
        result = torch.isin(elements, test_elements)
        return result
