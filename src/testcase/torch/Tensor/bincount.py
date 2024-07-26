import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.bincount)
class TorchTensorBincountTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_bincount_correctness(self):
        # Randomly generate the size of the input tensor
        num_elements = random.randint(1, 10)

        # Randomly generate the input tensor with integer values
        input_tensor = torch.randint(0, 10, (num_elements,))

        # Randomly generate weights tensor with the same size as input_tensor
        weights = torch.randn(num_elements)

        # Randomly generate minlength
        minlength = random.randint(0, 20)

        # Compute the bincount
        result = input_tensor.bincount(weights=weights, minlength=minlength)
        return result
