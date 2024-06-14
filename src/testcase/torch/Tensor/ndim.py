import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.ndim)
class TorchTensorNdimTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_ndim_correctness(self):
        """Test the correctness of torch.Tensor.ndim."""
        dim = random.randint(1, 4)  # Random dimension for the tensor
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]
        tensor = torch.randn(input_size)  # Generate a random tensor with the specified dimensions
        result = tensor.ndim  # Get the number of dimensions of the tensor
        return result
    
    
    
    