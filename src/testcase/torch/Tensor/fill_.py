import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.fill_)
class TorchTensorFillTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_fill_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensor
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]
        input_tensor = torch.randn(input_size)
        value = random.uniform(-10.0, 10.0)  # Random fill value between -10.0 and 10.0
        result = input_tensor.fill_(value)
        return result
