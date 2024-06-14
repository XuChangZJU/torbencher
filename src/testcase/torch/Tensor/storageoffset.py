import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.storageoffset)
class TorchTensorStorageoffsetTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_storage_offset_correctness(self):
    # Randomly generate tensor dimension and size
    dim = random.randint(1, 4)
    num_of_elements_each_dim = random.randint(1, 5)
    input_size = [num_of_elements_each_dim for i in range(dim)]

    # Create a random tensor
    tensor = torch.randn(input_size)

    # Randomly select a starting index for slicing
    start_index = random.randint(0, len(tensor) - 1) 

    # Slice the tensor
    sliced_tensor = tensor[start_index:]

    # Calculate the expected storage offset
    expected_offset = start_index

    # Get the actual storage offset
    result = sliced_tensor.storage_offset()
    
    return result
