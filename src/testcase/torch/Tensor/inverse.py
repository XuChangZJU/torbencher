import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.inverse)
class TorchTensorInverseTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_inverse_correctness(self):
        """
        Test the correctness of torch.Tensor.inverse with small scale random parameters.
        """
        # Generate random dimension and number of elements for the square matrix
        dim = random.randint(1, 4)
        input_size = [dim, dim]
    
        # Generate a random square matrix that is invertible
        matrix = torch.randn(input_size)
        while torch.det(matrix) == 0:  # Ensure the matrix is invertible
            matrix = torch.randn(input_size)
    
        # Calculate the inverse
        inverse_matrix = matrix.inverse()
    
        # Return the inverse matrix
        return inverse_matrix
    
    
    
    