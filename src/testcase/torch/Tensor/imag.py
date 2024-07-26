import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.imag)
class TorchTensorImagTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_imag_correctness(self):
        # Random dimension for the tensors
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Generate random input size
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate random complex tensor
        tensor = torch.randn(input_size, dtype=torch.cfloat)

        # Get imaginary part of the tensor
        result = tensor.imag
        return result
