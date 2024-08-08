import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.linalg.multi_dot)
class TorchLinalgMultiUdotTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_multi_dot_correctness(self):
        # Define the number of matrices to multiply
        num_matrices = random.randint(2, 5)

        # Generate random dimensions for the matrices, ensuring valid matrix multiplication
        matrix_dims = []
        for i in range(num_matrices + 1):
            if i == 0:
                # First dimension can be 1D or 2D
                dim = random.randint(1, 5)
                matrix_dims.append(dim)
            elif i == num_matrices:
                # Last dimension needs to match the second to last
                matrix_dims.append(matrix_dims[-1])
            else:
                # Intermediate dimensions need to match for valid multiplication
                matrix_dims.append(random.randint(1, 5))

        # Create the list of random tensors
        tensors = []
        for i in range(num_matrices):
            if i == 0 and len(matrix_dims) == 2:
                # First tensor can be 1D
                tensors.append(torch.randn(matrix_dims[i]))
            elif i == num_matrices - 1 and len(matrix_dims) == 2:
                # Last tensor can be 1D
                tensors.append(torch.randn(matrix_dims[i]))
            else:
                tensors.append(torch.randn(matrix_dims[i], matrix_dims[i + 1]))

        # Calculate the multi_dot result
        result = torch.linalg.multi_dot(tensors)

        return result
