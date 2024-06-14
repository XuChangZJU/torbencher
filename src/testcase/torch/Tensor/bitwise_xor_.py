import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.bitwise_xor_)
class TorchTensorBitwisexorTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_bitwise_xor__correctness(self):
        # Randomly generate input tensor size
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
    
        # Generate random input tensors
        input_tensor = torch.randint(0, 10, input_size)  # Generate integers between 0 and 9 (inclusive) to represent bits
        other_tensor = torch.randint(0, 10, input_size)  # Generate integers between 0 and 9 (inclusive) to represent bits
    
        # Perform in-place bitwise XOR operation
        input_tensor.bitwise_xor_(other_tensor)
    
        return input_tensor  # Return the modified tensor to observe the in-place effect
    
    
    
    