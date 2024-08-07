import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.utils.clip_grad_norm)
class TorchNnUtilsClipUgradUnormTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_clip_grad_norm_correctness(self):
        # Randomly generate the number of parameters
        num_params = random.randint(1, 5)

        # Randomly generate the size of each parameter tensor
        param_sizes = [[random.randint(1, 5) for _ in range(random.randint(1, 4))] for _ in range(num_params)]

        # Create a list of parameters with random values
        params = [torch.randn(size) for size in param_sizes]

        # Randomly generate a max_norm value
        max_norm = random.uniform(0.1, 10.0)

        # Clip the gradient norm of the parameters
        total_norm = torch.nn.utils.clip_grad_norm(params, max_norm)

        return total_norm
