import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.functional.upsample_nearest)
class TorchNnFunctionalUpsampleUnearestTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_upsample_nearest_correctness(self):
        # Random input size
        dim = random.randint(4, 5)  # Spatial and volumetric upsampling are supported
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Random input tensor
        input_tensor = torch.randn(input_size)

        # Randomly choose between 'size' and 'scale_factor'
        choice = random.choice(['size', 'scale_factor'])

        if choice == 'size':
            # Random output size
            output_size = [random.randint(1, 10) for _ in range(dim - 2)]  # Spatial dimensions
            result = torch.nn.functional.upsample_nearest(input_tensor, output_size)
        else:  # choice == 'scale_factor'
            # Random scale factor
            scale_factor = random.randint(1, 5)  # Has to be an integer
            result = torch.nn.functional.upsample_nearest(input_tensor, scale_factor=scale_factor)

        return result
