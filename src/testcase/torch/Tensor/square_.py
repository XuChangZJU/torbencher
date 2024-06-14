import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.square_)
class TorchTensorSquareTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_square__correctness(self):
        # Generate random dimension and size for the tensor
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
    
        # Generate a random tensor 
        input_tensor = torch.randn(input_size)
    
        # Perform in-place square operation
        input_tensor.square_()
    
        return input_tensor
    
    
    
    