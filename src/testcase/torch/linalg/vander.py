import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.linalg.vander)
class TorchLinalgVanderTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_vander_correctness(self):
        # Randomly generate the size of the input tensor
        num_elements = random.randint(2, 10)  # Ensure at least 2 elements for meaningful Vandermonde matrix
        input_tensor = torch.randn(num_elements)

        # Randomly decide whether to specify N or not
        if random.choice([True, False]):
            N = random.randint(1, 10)  # Random number of columns in the output
            result = torch.linalg.vander(input_tensor, N=N)
        else:
            result = torch.linalg.vander(input_tensor)

        return result
