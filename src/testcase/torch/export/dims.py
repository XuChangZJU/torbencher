import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.export.dims)
class TorchExportDimsTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_export_dims_correctness(self):
        # Randomly generate the number of dimensions to create
        num_dims = random.randint(1, 5)

        # Create a list of random dimensions
        dims = [torch.export.Dim(name=f"dim_{i}") for i in range(num_dims)]

        return dims
