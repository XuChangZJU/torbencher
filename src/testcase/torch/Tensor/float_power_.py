import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.float_power_)
class TorchTensorFloatUpowerUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_float_power__correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensor
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for _ in range(dim)]  # Generate input size for the tensor

        base_tensor = torch.randn(input_size, dtype=torch.double)  # Random base tensor with double precision
        exponent_tensor = torch.randn(input_size, dtype=torch.double)  # Random exponent tensor with double precision

        result = base_tensor.float_power_(exponent_tensor)  # In-place float power operation
        return result
