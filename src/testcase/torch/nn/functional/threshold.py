import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.threshold)
class TorchNnFunctionalThresholdTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_threshold_correctness(self):
        # Randomly generate tensor dimensions and size
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
    
        # Generate random input tensor
        input_tensor = torch.randn(input_size)
        # Generate random threshold value
        threshold_value = random.uniform(-1.0, 1.0)  # Threshold value can be any float
        # Generate random value to replace
        value = random.uniform(-5.0, 5.0)  # Value can be any float
    
        # Apply threshold operation
        result = torch.nn.functional.threshold(input_tensor, threshold_value, value)
        return result
    