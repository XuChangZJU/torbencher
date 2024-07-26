import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.corrcoef)
class TorchTensorCorrcoefTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_corrcoef_correctness(self):
        num_of_elements = random.randint(2, 10)  # Random number of elements for the tensor
        tensor_size = (num_of_elements, num_of_elements)  # Ensure the tensor is 2D for corrcoef

        tensor = torch.randn(tensor_size)
        result = torch.corrcoef(tensor)
        return result
