import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.narrow)
class TorchTensorNarrowTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_narrow_correctness(self):
        # Random dimension for the tensor
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Random input size
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Generate a random tensor with the specified input size
        input_tensor = torch.randn(input_size)
        # Randomly select a dimension to narrow
        dimension = random.randint(0, len(input_size) - 1)
        # Randomly select a start index for narrowing, ensuring it's within the dimension's bounds
        start = random.randint(0, input_size[dimension] - 1)
        # Randomly select a length for narrowing, ensuring it doesn't exceed the dimension's bounds
        length = random.randint(1, input_size[dimension] - start)
        # Apply the narrow operation
        result = input_tensor.narrow(dimension, start, length)
        # Return the result tensor
        return result
    
    
    
    