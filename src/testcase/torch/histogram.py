import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.histogram)
class TorchHistogramTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_histogram_correctness_with_random_input_tensor_and_bins_as_integer(self):
        dim = random.randint(1, 4)  # Random dimension for the tensor
        num_of_elements_each_dim = random.randint(1,5) # Random number of elements each dimension
        input_size=[num_of_elements_each_dim for i in range(dim)] 
    
        input_tensor = torch.randn(input_size)
        bins = random.randint(1, 10)  # Random number of bins
        hist, bin_edges = torch.histogram(input_tensor, bins)
        return hist, bin_edges
    