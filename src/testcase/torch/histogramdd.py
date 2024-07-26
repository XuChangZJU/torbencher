import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.histogramdd)
class TorchHistogramddTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_histogramdd_correctness(self):
        # Random number of dimensions (between 1 and 4), ensuring at least 2 dimensions for input tensor
        num_of_points = random.randint(1, 4)
        dimension = 2
    
        # Random number of points in each dimension (between 1 and 5)
        input_size = [num_of_points, dimension]
    
        # Generate random input tensor of specified size
        input_tensor = torch.randn(input_size)
    
        # Random number of bins for each dimension (between 1 and 5 bins per dimension)
        bins = [random.randint(1, 5) for _ in range(dimension)]
    
        # Compute histogram
        hist, bin_edges = torch.histogramdd(input_tensor, bins)
        
        return hist, bin_edges
    
    
    
    