import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.normal_)
class TorchTensorNormalUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_normal__correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]

        input_tensor = torch.randn(input_size)
        mean = random.uniform(-10.0, 10.0)  # Random mean value between -10.0 and 10.0
        std = random.uniform(0.1, 10.0)  # Random std value between 0.1 and 10.0 (std should be positive)
        result = input_tensor.normal_(mean, std)
        return result
