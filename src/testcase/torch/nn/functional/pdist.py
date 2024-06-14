import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.pdist)
class TorchNnFunctionalPdistTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_pdist_correctness(self):
        # Randomly generate the number of row vectors (N) and the number of elements in each row vector (M)
        num_rows = random.randint(2, 5)  # N should be at least 2 to compute pairwise distances
        num_elements = random.randint(1, 5)  # Random number of elements in each row vector
    
        # Generate a random input tensor of shape (N, M)
        input_tensor = torch.randn(num_rows, num_elements)
    
        # Randomly generate the p value for the p-norm distance
        p_value = random.uniform(0, 10)  # p value between 0 and 10
    
        # Compute the pairwise p-norm distances using torch.nn.functional.pdist
        result = torch.nn.functional.pdist(input_tensor, p=p_value)
        return result
    