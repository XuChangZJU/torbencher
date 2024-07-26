import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.from_dlpack)
class TorchFromdlpackTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_from_dlpack_correctness(self):
        # from_dlpack support directly convert tensor since 1.10.0
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]

        input_tensor = torch.randn(input_size)
        result = torch.from_dlpack(input_tensor)
        # modify result to check if it shares memory with input_tensor
        result[:] = 0
        return result
