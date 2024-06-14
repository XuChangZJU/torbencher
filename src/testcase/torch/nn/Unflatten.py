import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.Unflatten)
class TorchNnUnflattenTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_unflatten_correctness(self):
        # Randomly choose the dimension to unflatten
        dim = random.randint(0, 3)
        
        # Randomly generate the size of the input tensor
        input_size = [random.randint(2, 5) for _ in range(4)]
        
        # Ensure the size at the chosen dimension is a product of the unflattened size
        unflattened_size = [random.randint(2, 5) for _ in range(2)]
        input_size[dim] = unflattened_size[0] * unflattened_size[1]
        
        # Generate a random input tensor with the specified size
        input_tensor = torch.randn(input_size)
        
        # Create the Unflatten module
        unflatten = torch.nn.Unflatten(dim, unflattened_size)
        
        # Apply the Unflatten module to the input tensor
        result = unflatten(input_tensor)
        
        return result
    