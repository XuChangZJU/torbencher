import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.masked_select)
class TorchMaskedUselectTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_masked_select_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]

        input_tensor = torch.randn(input_size)
        # Generate a random mask tensor with the same size as input_tensor, ensuring broadcastable
        mask_tensor = torch.randint(0, 2, input_size, dtype=torch.bool)
        result = torch.masked_select(input_tensor, mask_tensor)
        return result
