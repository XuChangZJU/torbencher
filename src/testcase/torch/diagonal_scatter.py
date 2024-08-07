import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.diagonal_scatter)
class TorchDiagonalUscatterTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_diagonal_scatter_correctness(self):
        # Randomly generate input tensor size
        dim1 = random.randint(2, 5)  # input tensor must be at least 2-dimensional
        dim2 = random.randint(2, 5)
        dim3 = random.randint(1, 3)
        dim4 = random.randint(1, 3)
        input_size = [dim1, dim2, dim3, dim4]

        # Generate random input tensor
        input_tensor = torch.randn(input_size)

        # Generate random offset
        offset = random.randint(-(dim1 - 1), dim2 - 1)  # Ensure offset is within valid range

        # Calculate the size of the diagonal based on offset
        if offset >= 0:
            diagonal_size = min(dim1, dim2 - offset)
        else:
            diagonal_size = min(dim1 + offset, dim2)

        # Generate random src tensor with the correct size
        src_size = [dim3, dim4, diagonal_size]
        src_tensor = torch.randn(src_size)

        # Apply diagonal_scatter
        result = torch.diagonal_scatter(input_tensor, src_tensor, offset)
        return result
