import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.linalg.svdvals)
class TorchLinalgSvdvalsTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_torch_linalg_svdvals_correctness(self):
        # Define the dimension of the input tensor
        dim = random.randint(1, 4)
        # Define the size of each dimension of the input tensor
        m = random.randint(1, 5)
        n = random.randint(1, 5)
        # Generate a random input tensor of shape (*, m, n)
        input_tensor = torch.randn([m, n] if dim == 1 else [random.randint(1, 5) for _ in range(dim - 1)] + [m, n])
        # Calculate the singular values of the input tensor
        result = torch.linalg.svdvals(input_tensor)
        # Return the calculated singular values
        return result
