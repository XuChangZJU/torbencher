import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cartesianprod)
class TorchCartesianprodTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_cartesian_prod_correctness(self):
    # Randomly generate the number of tensors
    num_of_tensors = random.randint(2, 5)
    
    # Randomly generate the size of each tensor
    tensor_sizes = [random.randint(1, 5) for _ in range(num_of_tensors)]
    
    # Generate random tensors
    tensors = [torch.randn(size) for size in tensor_sizes]
    
    # Calculate the cartesian product
    result = torch.cartesian_prod(*tensors)
    return result
