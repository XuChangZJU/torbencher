import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.dot)
class TorchTensorDotTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_dot_correctness(self):
    # Random dimension for the tensors (1D tensors for dot product)
    dim = random.randint(1, 10)  # Random dimension size between 1 and 10

    # Generate random 1D tensors of the same size
    tensor1 = torch.randn(dim)
    tensor2 = torch.randn(dim)

    # Perform dot product
    result = tensor1.dot(tensor2)
    return result
