import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.pinmemory)
class TorchTensorPinmemoryTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_pin_memory_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensor
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for _ in range(dim)]  # Generate input size list
    
        tensor = torch.randn(input_size)  # Create a random tensor
        pinned_tensor = tensor.pin_memory()  # Pin the tensor to memory
    
        return pinned_tensor.is_pinned()  # Check if the tensor is pinned
    