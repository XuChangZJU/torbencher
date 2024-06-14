import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.unflatten)
class TorchTensorUnflattenTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_unflatten_correctness(self):
    # Randomly generate dimensions for the original tensor
    original_dim = random.randint(2, 4)
    num_of_elements_each_dim = random.randint(2, 5)
    input_size = [num_of_elements_each_dim for _ in range(original_dim)]
    
    # Create a random tensor with the generated dimensions
    tensor = torch.randn(input_size)
    
    # Randomly choose a dimension to unflatten
    dim_to_unflatten = random.randint(0, original_dim - 1)
    
    # Generate sizes for unflattening
    size1 = random.randint(2, 5)
    size2 = tensor.size(dim_to_unflatten) // size1
    while tensor.size(dim_to_unflatten) % size1 != 0:
        size1 = random.randint(2, 5)
        size2 = tensor.size(dim_to_unflatten) // size1
    
    sizes = (size1, size2)
    
    # Perform the unflatten operation
    result = tensor.unflatten(dim_to_unflatten, sizes)
    return result
