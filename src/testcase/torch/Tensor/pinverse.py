import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.pinverse)
class TorchTensorPinverseTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_pinverse_correctness(self):
    # Random dimension for the tensor (2D matrix)
    rows = random.randint(2, 5)  # Random number of rows
    cols = random.randint(2, 5)  # Random number of columns

    # Generate a random 2D tensor (matrix)
    matrix = torch.randn(rows, cols)

    # Compute the pseudo-inverse of the matrix
    result = matrix.pinverse()
    return result
