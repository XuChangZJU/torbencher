import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.bitwise_or)
class TorchTensorBitwiseorTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_bitwise_or_correctness(self):
        # Define the dimension and size of the tensors randomly
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate random tensors of type long
        tensor1 = torch.randint(0, 10, input_size,
                                dtype=torch.long)  # generate random tensor with element values between 0 and 9
        tensor2 = torch.randint(0, 10, input_size,
                                dtype=torch.long)  # generate random tensor with element values between 0 and 9

        result = tensor1.bitwise_or(tensor2)
        return result
