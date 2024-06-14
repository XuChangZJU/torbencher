import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.asstrided)
class TorchTensorAsstridedTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_as_strided_correctness(self):
    # Randomly generate the size of the original tensor
    original_dim = random.randint(1, 4)
    original_num_of_elements_each_dim = random.randint(1, 5)
    original_size = [original_num_of_elements_each_dim for _ in range(original_dim)]
    
    # Create the original tensor with random values
    original_tensor = torch.randn(original_size)
    
    # Randomly generate the size of the new tensor
    new_dim = random.randint(1, 4)
    new_num_of_elements_each_dim = random.randint(1, 5)
    new_size = [new_num_of_elements_each_dim for _ in range(new_dim)]
    
    # Randomly generate the stride for the new tensor
    stride = [random.randint(1, 3) for _ in range(new_dim)]
    
    # Randomly generate the storage offset
    storage_offset = random.randint(0, original_tensor.numel() - 1)
    
    # Apply as_strided to the original tensor
    result = original_tensor.as_strided(new_size, stride, storage_offset)
    return result
