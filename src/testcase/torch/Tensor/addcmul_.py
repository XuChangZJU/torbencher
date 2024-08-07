import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.addcmul_)
class TorchTensorAddcmulUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_addcmul__correctness(self):
        # Randomly generate tensor dimensions and number of elements
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate random tensors of the same size
        tensor1 = torch.randn(input_size)
        tensor2 = torch.randn(input_size)
        tensor3 = torch.randn(input_size)

        # Perform addcmul_ operation
        tensor1.addcmul_(tensor2, tensor3)  # tensor1 is modified in-place

        return tensor1  # Return modified tensor1 to check the effect
