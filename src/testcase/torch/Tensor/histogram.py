import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.histogram)
class TorchTensorHistogramTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_torch_Tensor_histogram_correctness(self):
        # Define the dimension of the input tensor
        dim = random.randint(1, 4)
        # Define the number of elements in each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Generate the input size
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Generate a random tensor
        input_tensor = torch.randn(input_size)
        # Generate a random number of bins
        bins = random.randint(1, 10)
        # Calculate the histogram
        hist, bin_edges = input_tensor.histogram(bins)
        # Return the histogram and bin edges
        return hist, bin_edges
