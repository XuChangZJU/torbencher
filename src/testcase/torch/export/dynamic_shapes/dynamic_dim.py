import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.export.dynamic_shapes.dynamic_dim)
class TorchExportDynamicUshapesDynamicUdimTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_dynamic_dim_correctness(self):
        # Randomly generate dimensions and size for the tensor
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]

        # Create a random tensor with the generated size
        tensor = torch.randn(input_size)

        # Randomly select a dimension to be dynamic
        dynamic_dim = random.randint(0, dim - 1)

        # Apply dynamic_dim to the selected dimension
        # Assuming the correct function is torch.export.dynamic_dim
        result = torch.export.dynamic_dim(tensor, dynamic_dim)
        return result
