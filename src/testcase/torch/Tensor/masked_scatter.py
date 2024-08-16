import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.masked_scatter)
class TorchTensorMaskedUscatterTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_masked_scatter_correctness(self):
        # Define the dimension and size of the tensors
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate random tensors
        self_tensor = torch.randn(input_size)
        # Generate a random mask with the same shape as self_tensor
        mask = torch.randint(0, 2, input_size, dtype=torch.bool)
        # Generate a source tensor with shape broadcastable to self_tensor where mask is True
        source_tensor = torch.randn(mask.sum().item())

        # Apply masked_scatter
        result = self_tensor.masked_scatter(mask, source_tensor)

        return result
