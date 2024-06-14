import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.diag)
class TorchDiagTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_diag_correctness_vector(self):
        # Randomly generate the size of the vector
        vector_length = random.randint(1, 10)
        
        # Generate a random vector of the given size
        vector = torch.randn(vector_length)
        
        # Compute the diagonal matrix from the vector
        result = torch.diag(vector)
        return result
    