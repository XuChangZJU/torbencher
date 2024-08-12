import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.all)
class TorchTensorAllTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_all_correctness(self):
        # Random dimension for the tensors
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Generate random size for the input tensor
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Generate random tensor data
        tensor = torch.randn(input_size)
        # Call torch.Tensor.all()
        result = tensor.all()
        return result
