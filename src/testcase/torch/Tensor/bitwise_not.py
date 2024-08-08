import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.bitwise_not)
class TorchTensorBitwiseUnotTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_bitwise_not_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensor
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for _ in range(dim)]

        tensor = torch.randint(0, 256, input_size,
                               dtype=torch.int32)  # Random tensor with integer values between 0 and 255
        result = tensor.bitwise_not()
        return result
