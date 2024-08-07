import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.polygamma_)
class TorchTensorPolygammaUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_polygamma__correctness(self):
        # Random input size
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate random input tensor
        input_tensor = torch.randn(input_size)

        # Generate random 'n' for polygamma function (n should be a positive integer)
        n = random.randint(1, 5)

        # Apply polygamma_ function
        input_tensor.polygamma_(n)

        return input_tensor
