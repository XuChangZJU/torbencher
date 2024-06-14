import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.max)
class TorchTensorMaxTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_torch_Tensor_max_correctness(self):
        # Define the dimension of the tensor
        dim = random.randint(1, 4)
        # Define the number of elements in each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Create the input size list
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Generate a random tensor with the specified input size
        random_tensor = torch.randn(input_size)
        # Calculate the max value and its index along all dimensions
        max_value = random_tensor.max()
        # Return the max value
        return max_value
    
    def test_torch_Tensor_max_dim_correctness(self):
        # Define the dimension of the tensor
        dim = random.randint(1, 4)
        # Define the number of elements in each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Create the input size list
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Generate a random tensor with the specified input size
        random_tensor = torch.randn(input_size)
        # Randomly select a dimension
        dim = random.randint(0, len(input_size) - 1)
        # Calculate the max value and its index along the specified dimension
        max_value, max_index = random_tensor.max(dim)
        # Return the max value and its index
        return max_value, max_index
    
    