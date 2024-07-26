import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.copy_)
class TorchTensorCopyTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_copy_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]

        src_tensor = torch.randn(input_size)
        # Create self tensor with the same size as src_tensor to ensure broadcastability
        self_tensor = torch.randn(input_size)
        result = self_tensor.copy_(src_tensor)
        return result
