import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.ger)
class TorchTensorGerTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_ger_correctness(self):
        # Generate random dimensions for the input tensors
        dim1 = random.randint(1, 5)
        dim2 = random.randint(1, 5)

        # Create random input tensors
        tensor1 = torch.randn(dim1)  # 1D tensor
        tensor2 = torch.randn(dim2)  # 1D tensor

        # Calculate the outer product using torch.Tensor.ger
        result = tensor1.ger(tensor2)

        # Return the result tensor
        return result
