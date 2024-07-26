import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.bmm)
class TorchBmmTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_bmm_correctness(self):
        # Randomly generate the batch size, matrix dimensions
        batch_size = random.randint(1, 10)
        n = random.randint(1, 10)
        m = random.randint(1, 10)
        p = random.randint(1, 10)

        # Create input tensors with the specified dimensions
        input_tensor = torch.randn(batch_size, n, m)
        mat2_tensor = torch.randn(batch_size, m, p)  # Ensure valid matrix multiplication

        # Perform batch matrix multiplication
        result = torch.bmm(input_tensor, mat2_tensor)
        return result
