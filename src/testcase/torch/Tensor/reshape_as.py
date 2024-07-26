import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.reshape_as)
class TorchTensorReshapeasTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_reshape_as_correctness(self):
        # Random dimension for the tensors
        dim = random.randint(1, 4)

        # Random number of elements each dimension for the original tensor
        num_of_elements_each_dim = random.randint(1, 5)
        original_size = [num_of_elements_each_dim for _ in range(dim)]

        # Random number of elements each dimension for the target tensor
        target_size = [num_of_elements_each_dim for _ in range(dim)]

        # Generate random tensors
        original_tensor = torch.randn(original_size)
        target_tensor = torch.randn(target_size)

        # Reshape original tensor to the shape of target tensor
        reshaped_tensor = original_tensor.reshape_as(target_tensor)

        return reshaped_tensor
