import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.geometric_)
class TorchTensorGeometricUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_geometric_correctness(self):
        dim = 4  # Random dimension for the tensors
        num_of_elements_each_dim = 5  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]
        input_tensor = torch.randn(input_size)
        p = random.uniform(0.1, 1.0)  # Random p value between 0.1 and 1.0
        result = input_tensor.geometric_(p)
        return result.shape
