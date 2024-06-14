import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.normal)
class TorchNormalTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_normal_correctness(self):
    # Randomly generate the shape of the mean tensor
    mean_dim = random.randint(1, 4)  # Random dimension for the mean tensor
    num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
    mean_size = [num_of_elements_each_dim for _ in range(mean_dim)]
    
    # Generate mean tensor with random values
    mean_tensor = torch.randn(mean_size)
    
    # Generate std tensor with the same number of elements but different shape
    std_dim = random.randint(1, 4)  # Random dimension for the std tensor
    num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
    std_size = [num_of_elements_each_dim for _ in range(std_dim)]
    std_tensor = torch.randn(std_size).abs()  # Standard deviation tensor should be positive

    # Drawing random samples from normal distribution
    result = torch.normal(mean_tensor, std_tensor)
    
    return result
