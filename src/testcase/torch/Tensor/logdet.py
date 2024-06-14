import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.logdet)
class TorchTensorLogdetTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_logdet_correctness(self):
    # Generate a random square matrix
    dim = random.randint(1, 10)  # Random dimension for the matrix
    input_size = [dim, dim] 

    # Generate a random tensor
    tensor = torch.randn(input_size)

    # Calculate the log determinant
    result = tensor.logdet()
    return result
