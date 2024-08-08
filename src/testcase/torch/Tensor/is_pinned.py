import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.is_pinned)
class TorchTensorIsUpinnedTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_is_pinned_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensor
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for _ in range(dim)]  # Generate input size list

        tensor = torch.randn(input_size)  # Create a random tensor

        # Pin the tensor to memory only if it's on the CPU
        if tensor.device.type == 'cpu':
            pinned_tensor = tensor.pin_memory()  # Pin the tensor to memory
            result_pinned = pinned_tensor.is_pinned()  # Check if the pinned tensor is pinned
        else:
            result_pinned = False  # If tensor is not on CPU, it cannot be pinned

        result_unpinned = tensor.is_pinned()  # Check if the original tensor is pinned

        return result_unpinned, result_pinned
