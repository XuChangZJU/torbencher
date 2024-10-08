import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


def swap_tensors(tensor1, tensor2):
    return tensor2, tensor1


@test_api(torch.utils.swap_tensors)
class TorchUtilsSwapUtensorsTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_swap_tensors_correctness(self):
        # Random dimension for the tensors
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]

        # Generate two random tensors with the same shape
        tensor1 = torch.randn(input_size)
        tensor2 = torch.randn(input_size)

        # Swap the tensors
        swapped_tensor1, swapped_tensor2 = swap_tensors(tensor1, tensor2)

        return swapped_tensor1, swapped_tensor2
