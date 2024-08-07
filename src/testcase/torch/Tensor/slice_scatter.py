import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.slice_scatter)
class TorchTensorSliceUscatterTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_slice_scatter_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(2, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for _ in range(dim)]

        tensor = torch.randn(input_size)

        # Randomly choose a dimension to slice along
        slice_dim = random.randint(0, dim - 1)

        # Randomly choose start and end indices for slicing
        start = random.randint(0, num_of_elements_each_dim - 1)
        end = random.randint(start + 1, num_of_elements_each_dim)

        # Randomly choose a step for slicing
        step = random.randint(1, num_of_elements_each_dim)

        # Calculate the size of the src tensor to match the slice size
        slice_size = list(input_size)
        slice_size[slice_dim] = (end - start + step - 1) // step

        src = torch.randn(slice_size)

        result = tensor.slice_scatter(src, slice_dim, start, end, step)
        return result
