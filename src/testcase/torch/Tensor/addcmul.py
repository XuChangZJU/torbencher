import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.addcmul)
class TorchTensorAddcmulTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_torch_Tensor_addcmul_correctness(self):
        # Define the dimension of the tensors
        dim = random.randint(1, 4)
        # Define the number of elements in each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Create the input size list
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Generate the input tensors
        input = torch.randn(input_size)
        tensor1 = torch.randn(input_size)
        tensor2 = torch.randn(input_size)
        # Calculate the result of the addcmul operation
        result = input.addcmul(tensor1, tensor2)
        # Return the result
        return result
    
    
    
    