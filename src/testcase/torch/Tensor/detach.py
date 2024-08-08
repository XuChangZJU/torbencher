import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.detach)
class TorchTensorDetachTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_detach_correctness(self):
        # Generate random dimension and size for the tensor
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Create a random tensor
        original_tensor = torch.randn(input_size, requires_grad=True)

        # Detach the tensor
        detached_tensor = original_tensor.detach()

        # Check if the detached tensor requires gradient
        assert not detached_tensor.requires_grad, "Detached tensor should not require gradient"

        # Check if the original and detached tensors share the same storage
        assert detached_tensor.storage().data_ptr() == original_tensor.storage().data_ptr(), "Detached tensor should share storage with the original tensor"

        return detached_tensor
