import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.Softmin)
class TorchNnSoftminTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_nn_Softmin_correctness(self):
    # Define the dimension of the input tensor
    dim = random.randint(1, 4)
    # Define the number of elements in each dimension
    num_of_elements_each_dim = random.randint(1, 5)
    # Create the input size list
    input_size = [num_of_elements_each_dim for i in range(dim)]
    # Generate a random tensor with the specified input size
    input_tensor = torch.randn(input_size)
    # Define a random dimension along which Softmin will be computed
    dim_to_compute = random.randint(0, len(input_size) - 1)
    # Create a Softmin module with the specified dimension
    softmin_module = torch.nn.Softmin(dim=dim_to_compute)
    # Apply the Softmin function to the input tensor
    output_tensor = softmin_module(input_tensor)
    # Return the output tensor
    return output_tensor
