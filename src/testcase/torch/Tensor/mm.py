import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.mm)
class TorchTensorMmTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_torch_Tensor_mm_correctness(self):
        # Generate random dimensions for the matrices
        dim1 = random.randint(1, 10)
        dim2 = random.randint(1, 10)
        dim3 = random.randint(1, 10)
        # Generate random input tensors
        input_size1 = [dim1, dim2]
        input_size2 = [dim2, dim3]  # dim2 should be same to make mm valid
        tensor1 = torch.randn(input_size1)
        tensor2 = torch.randn(input_size2)
        # Perform matrix multiplication
        result = tensor1.mm(tensor2)
        return result
