import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.normal)
class TorchNormalTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_normal_correctness(self):
        # Randomly generate the shape of the mean tensor
        mean_dim = random.randint(1, 4)  # Random dimension for the mean tensor
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        mean_size = [num_of_elements_each_dim for _ in range(mean_dim)]

        # Generate mean tensor with random values
        mean_tensor = torch.randn(mean_size)

        # Generate std tensor with the same shape as mean tensor
        std_tensor = torch.randn(mean_size).abs()  # Standard deviation tensor should be positive

        # Drawing random samples from normal distribution
        result = torch.normal(mean_tensor, std_tensor)

        return result

    def test_normal_with_shared_mean(self):
        # Shared mean value
        mean = random.uniform(-10.0, 10.0)

        # Randomly generate the shape of the std tensor
        std_dim = random.randint(1, 4)  # Random dimension for the std tensor
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        std_size = [num_of_elements_each_dim for _ in range(std_dim)]

        # Generate std tensor with random positive values
        std_tensor = torch.randn(std_size).abs()

        # Drawing random samples from normal distribution
        result = torch.normal(mean, std_tensor)

        return result

    def test_normal_with_shared_std(self):
        # Randomly generate the shape of the mean tensor
        mean_dim = random.randint(1, 4)  # Random dimension for the mean tensor
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        mean_size = [num_of_elements_each_dim for _ in range(mean_dim)]

        # Generate mean tensor with random values
        mean_tensor = torch.randn(mean_size)

        # Shared standard deviation value
        std = random.uniform(0.1, 10.0)

        # Drawing random samples from normal distribution
        result = torch.normal(mean_tensor, std)

        return result

    def test_normal_with_shared_mean_std(self):
        # Shared mean value
        mean = random.uniform(-10.0, 10.0)

        # Shared standard deviation value
        std = random.uniform(0.1, 10.0)

        # Randomly generate the shape of the output tensor
        size_dim = random.randint(1, 4)  # Random dimension for the output tensor
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        output_size = [num_of_elements_each_dim for _ in range(size_dim)]

        # Drawing random samples from normal distribution
        result = torch.normal(mean, std, size=output_size)

        return result
