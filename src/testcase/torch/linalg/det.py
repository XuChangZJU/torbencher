import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.linalg.det)
class TorchLinalgDetTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_linalg_det_correctness(self):
        # Generate a random dimension for the square matrix
        dim = random.randint(1, 4)
        # Generate a random number of elements for each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Create the input size list for the tensor
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Add the dimension of the square matrix to the input size list
        input_size.extend([dim, dim])
        # Create a random tensor of the specified input size
        A = torch.randn(input_size)
        # Calculate the determinant of the tensor
        result = torch.linalg.det(A)
        # Return the result
        return result
    
    
    
    