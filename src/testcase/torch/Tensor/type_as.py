import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.type_as)
class TorchTensorTypeUasTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_type_as_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]

        input_tensor = torch.randn(input_size)
        tensor_with_desired_type = torch.randn(
            input_size).double()  # Generate a tensor with desired type (double in this case)
        result = input_tensor.type_as(tensor_with_desired_type)
        return result
