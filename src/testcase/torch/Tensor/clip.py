import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.clip)
class TorchTensorClipTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_clip_correctness(self):
        # Randomly generate tensor dimensions and size
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
    
        # Generate a random tensor
        input_tensor = torch.randn(input_size)
    
        # Generate random min and max values for clipping
        min_val = random.uniform(-10.0, 0.0)  # Random min value between -10.0 and 0.0
        max_val = random.uniform(0.0, 10.0)  # Random max value between 0.0 and 10.0
    
        # Apply clip operation
        result = input_tensor.clip(min_val, max_val)
    
        return result
    
    
    
    