import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.tensordot)
class TorchTensordotTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_tensordot_correctness_with_small_scale_random_parameters(self):
    # Define the dimensions of the input tensors
    dim_a = random.randint(1, 3)
    dim_b = random.randint(1, 3)
    # Generate random sizes for each dimension
    size_a = [random.randint(1, 5) for _ in range(dim_a)]
    size_b = [random.randint(1, 5) for _ in range(dim_b)]
    
    # Ensure at least one dimension can be contracted
    num_contracted_dims = random.randint(1, min(dim_a, dim_b))
    # The sizes of the contracted dimensions must match
    for i in range(num_contracted_dims):
        size_b[i] = size_a[dim_a - num_contracted_dims + i]
    
    # Create the input tensors
    a = torch.randn(size_a)
    b = torch.randn(size_b)
    # Specify the dimensions to contract
    dims = ([list(range(dim_a - num_contracted_dims, dim_a)), list(range(num_contracted_dims))])
    result = torch.tensordot(a, b, dims)
    return result
