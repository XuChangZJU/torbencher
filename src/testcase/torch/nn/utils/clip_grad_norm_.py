import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.utils.clip_grad_norm_)
class TorchNnUtilsClipUgradUnormUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_clip_grad_norm_correctness(self):
        # Generate random dimension and number of elements for the tensor
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate a random tensor with gradients
        parameters = torch.randn(input_size, requires_grad=True)
        # Generate a random max_norm value
        max_norm = random.uniform(0.1, 10.0)
        # Calculate the gradients
        parameters.sum().backward()
        # Clip the gradients
        total_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm)

        return total_norm
