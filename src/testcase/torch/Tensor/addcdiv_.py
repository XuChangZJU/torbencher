import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.addcdiv_)
class TorchTensorAddcdivTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_torch_Tensor_addcdiv__correctness(self):
        # Define the dimension and size of the tensors
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate random tensors
        tensor1 = torch.randn(input_size)
        tensor2 = torch.randn(input_size)  # tensor1 and tensor2 should have the same size
        tensor3 = torch.randn(input_size)  # tensor1 and tensor3 should have the same size
        value = random.uniform(0.1, 10)  # value can be any random float

        # Perform the addcdiv_ operation
        tensor1.addcdiv_(tensor2, tensor3, value=value)

        return tensor1
