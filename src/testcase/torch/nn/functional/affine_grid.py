import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.functional.affine_grid)
class TorchNnFunctionalAffinegridTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_affine_grid_correctness(self):
        # Randomly choose between 2D and 3D affine grid
        is_3d = random.choice([True, False])

        # Random batch size
        N = random.randint(1, 10)

        if is_3d:
            # For 3D affine grid
            theta_shape = (N, 3, 4)
            output_size = (N, random.randint(1, 5), random.randint(1, 5), random.randint(1, 5), random.randint(1, 5))
        else:
            # For 2D affine grid
            theta_shape = (N, 2, 3)
            output_size = (N, random.randint(1, 5), random.randint(1, 5), random.randint(1, 5))

        # Generate random affine matrices
        theta = torch.randn(theta_shape)

        # Generate affine grid
        grid = torch.nn.functional.affine_grid(theta, output_size)

        return grid
