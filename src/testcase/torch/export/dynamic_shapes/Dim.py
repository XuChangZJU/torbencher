import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.export.dynamic_shapes.Dim)
class TorchExportDynamicshapesDimTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_dim_correctness(self):
        # Randomly generate a dimension size
        dim_size = random.randint(1, 10)

        # Create a random tensor with the generated dimension size
        tensor = torch.randn(dim_size)

        # Create a Dim object from the tensor's shape
        dim = torch._dynamo.dynamic_shapes.Dim(tensor.shape[0])

        # Return the dimension value
        return dim.value()
