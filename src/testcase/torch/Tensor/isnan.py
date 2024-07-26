import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.isnan)
class TorchTensorIsnanTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_isnan_correctness(self):
        """
        Test the correctness of torch.Tensor.isnan.
        """
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]

        tensor = torch.randn(input_size)
        # Replace some elements with NaN
        tensor[torch.rand(input_size) > 0.5] = float('nan')
        result = tensor.isnan()
        return result
