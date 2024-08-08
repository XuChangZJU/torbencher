import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.utils.clip_grad_value_)
class TorchNnUtilsClipUgradUvalueUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_clip_grad_value_correctness(self):
        # Random dimension for the tensor
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]

        # Generate a random tensor with gradients
        tensor = torch.randn(input_size, requires_grad=True)
        # Perform a dummy operation to create gradients
        tensor.sum().backward()

        # Random clip value between 0.1 and 10.0
        clip_value = random.uniform(0.1, 10.0)

        # Clip the gradients
        torch.nn.utils.clip_grad_value_(tensor, clip_value)

        # Return the clipped gradients
        return tensor.grad
