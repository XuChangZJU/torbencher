import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.floor_divide)
class TorchTensorFloorUdivideTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_floor_divide_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for _ in range(dim)]

        tensor1 = torch.randn(input_size) * 10  # Scale tensor to avoid division by zero
        tensor2 = torch.randn(input_size) * 10  # Scale tensor to avoid division by zero

        # Ensure tensor2 has no zero elements to avoid division by zero
        tensor2 = torch.where(tensor2 == 0, torch.ones_like(tensor2), tensor2)

        result = tensor1.floor_divide(tensor2)
        return result
