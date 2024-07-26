import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.unsqueeze_)
class TorchTensorUnsqueezeTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_unsqueeze__correctness(self):
        dim = random.randint(0, 3)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]

        input_tensor = torch.randn(input_size)
        dim = random.randint(-len(input_size) - 1, len(input_size))  # Random valid dim
        result = input_tensor.unsqueeze_(dim)
        return result
