import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.take_along_dim)
class TorchTakealongdimTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_take_along_dim_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]

        input_tensor = torch.randn(input_size)
        indices = torch.randint(0, input_tensor.size(dim=random.randint(0, len(input_size) - 1)),
                                size=input_size)  # Generate random indices within the valid range
        result = torch.take_along_dim(input_tensor, indices)
        return result
