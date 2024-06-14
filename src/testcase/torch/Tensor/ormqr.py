import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.ormqr)
class TorchTensorOrmqrTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_ormqr_correctness(self):
        # Randomly generate dimensions for the input tensors
        dim = random.randint(2, 4)  # Random dimension for the tensors (at least 2D for valid ormqr operation)
        num_of_elements_each_dim = random.randint(2, 5)  # Random number of elements each dimension
    
        # Generate random input tensors with appropriate sizes
        input_size = [num_of_elements_each_dim for _ in range(dim)]
        input2 = torch.randn(input_size)
        input3 = torch.randn(input_size)
    
        # Randomly choose left and transpose parameters
        left = random.choice([True, False])
        transpose = random.choice([True, False])
    
        # Perform the ormqr operation
        result = torch.Tensor.ormqr(input2, input3, left, transpose)
        return result
    