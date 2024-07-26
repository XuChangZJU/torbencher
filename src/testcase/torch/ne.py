import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.ne)
class TorchNeTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_ne_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]

        tensor1 = torch.randn(input_size)
        # Generate tensor2 with both equal and different elements compared to tensor1
        tensor2 = tensor1.clone()
        # Modify some elements of tensor2 to be different from tensor1
        mask = torch.randint(0, 2, tensor2.shape, dtype=torch.bool)
        tensor2[mask] = torch.randn_like(tensor2[mask])
        result = torch.ne(tensor1, tensor2)
        return result
