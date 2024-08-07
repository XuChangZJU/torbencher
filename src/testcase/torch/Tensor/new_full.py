import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.new_full)
class TorchTensorNewUfullTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_new_full_correctness(self):
        # Random dimension for the tensors
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Random size for the output tensor
        size = [num_of_elements_each_dim for i in range(dim)]
        # Random fill value
        fill_value = random.uniform(-10.0, 10.0)
        # Create a random tensor
        tensor = torch.randn([random.randint(1, 5) for i in range(dim)])
        # Call new_full
        result = tensor.new_full(size, fill_value)
        return result
