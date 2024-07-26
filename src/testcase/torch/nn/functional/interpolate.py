import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.interpolate)
class TorchNnFunctionalInterpolateTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_interpolate_correctness(self):
        dim = random.randint(3, 5)  # Random dimension for the tensors between 3 and 5
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]
        input_tensor = torch.randn(input_size)
    
        # Randomly choose between 'size' and 'scale_factor'
        if random.choice([True, False]):
            # Generate random size
            output_size = [random.randint(1, 10) for i in range(dim - 2)]  # spatial dimensions
            result = torch.nn.functional.interpolate(input_tensor, output_size)
        else:
            # Generate random scale_factor
            scale_factor = [random.uniform(0.1, 10.0) for i in range(dim - 2)]  # spatial dimensions
            result = torch.nn.functional.interpolate(input_tensor, scale_factor=scale_factor)
        return result
    
    
    
    