import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.functional.interpolate)
class TorchNnFunctionalInterpolateTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_interpolate_correctness(self):
        dim = random.randint(3, 5)  # Random dimension for the tensors between 3 and 5
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]
        input_tensor = torch.randn(input_size)

        # Randomly choose between 'size' and 'scale_factor'
        if random.choice([True, False]):
            # Generate random size ensuring no dimension is zero
            output_size = []
            while len(output_size) < dim - 2 or any(s == 0 for s in output_size):
                output_size = [random.randint(1, 10) for i in range(dim - 2)]
            result = torch.nn.functional.interpolate(input_tensor, output_size)
        else:
            # Generate random scale_factor ensuring no dimension results in zero
            scale_factor = []
            while len(scale_factor) < dim - 2 or any(sf <= 0 for sf in scale_factor):
                scale_factor = [random.uniform(0.1, 10.0) for i in range(dim - 2)]
            # Calculate output size to ensure it's valid
            output_size = [int(size * sf) for size, sf in zip(input_size[2:], scale_factor)]
            while any(s <= 0 for s in output_size):
                scale_factor = [random.uniform(0.1, 10.0) for i in range(dim - 2)]
                output_size = [int(size * sf) for size, sf in zip(input_size[2:], scale_factor)]
            result = torch.nn.functional.interpolate(input_tensor, scale_factor=scale_factor)
        return result
