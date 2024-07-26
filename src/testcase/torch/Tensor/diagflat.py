import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.diagflat)
class TorchTensorDiagflatTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_diagflat_correctness(self):
        # Random dimension for the tensor
        dim = random.randint(1, 4)
        # Random number of elements for the tensor
        num_of_elements = random.randint(1, 5)
        # Generate a random 1D tensor
        input_tensor = torch.randn(num_of_elements)
        # Random offset
        offset = random.randint(-num_of_elements,
                                num_of_elements)  # offset should be within the range of -num_of_elements and num_of_elements
        result = input_tensor.diagflat(offset)
        return result
