import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.module_load)
class TorchTensorModuleloadTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_module_load_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for _ in range(dim)]

        self_tensor = torch.randn(input_size)
        other_tensor = torch.randn(input_size)
        assign = random.choice([True, False])  # Randomly choose between True and False for assign

        if assign:
            self_tensor.copy_(other_tensor)
            result = self_tensor
        else:
            result = other_tensor

        return result
