import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.vstack)
class TorchVstackTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_vstack_correctness(self):
        # Random number of tensors to stack
        num_tensors = random.randint(2, 5)

        # Randomly determine the number of columns, ensuring all tensors have the same number of columns
        num_columns = random.randint(1, 4)

        # Random small scale dimensions for each tensor
        tensors = []
        for _ in range(num_tensors):
            num_rows = random.randint(1, 4)
            tensors.append(torch.randn(num_rows, num_columns))
            # size = [random.randint(1, 4) for _ in range(2)]  # Ensure 2D tensors for vstack
            # tensors.append(torch.randn(size))

        result = torch.vstack(tensors)

        # Check the shape of the resulting tensor
        expected_rows = sum(tensor.size(0) for tensor in tensors)
        expected_shape = (expected_rows, num_columns)

        assert result.shape == expected_shape, f"Expected shape {expected_shape}, but got {result.shape}"

        return result
