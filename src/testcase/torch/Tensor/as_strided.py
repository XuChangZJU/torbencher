import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.as_strided)
class TorchTensorAsUstridedTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_as_strided_correctness(self):
        # Randomly generate the size of the original tensor
        original_dim = random.randint(1, 4)
        original_num_of_elements_each_dim = random.randint(1, 5)
        original_size = [original_num_of_elements_each_dim for _ in range(original_dim)]

        # Create the original tensor with random values
        original_tensor = torch.randn(original_size)
        max_elements = original_tensor.numel()

        # Randomly generate the size and stride for the new tensor ensuring constraints
        new_dim = random.randint(1, 4)
        remaining_elements = max_elements
        new_size = []
        stride = []

        for i in range(new_dim):
            max_size = remaining_elements
            size = random.randint(1, max_size)
            new_size.append(size)

            if size > 1:
                max_stride = max(1, (remaining_elements - 1) // (size - 1))
            else:
                max_stride = remaining_elements
            stride_value = random.randint(1, min(3, max_stride))
            stride.append(stride_value)

            remaining_elements -= (size - 1) * stride_value

        # Ensure the storage offset is within bounds
        required_storage_size = sum((new_size[i] - 1) * stride[i] for i in range(new_dim)) + 1
        storage_offset = random.randint(0, max(0, max_elements - required_storage_size))

        # Apply as_strided to the original tensor
        result = original_tensor.as_strided(new_size, stride, storage_offset)
        return result
