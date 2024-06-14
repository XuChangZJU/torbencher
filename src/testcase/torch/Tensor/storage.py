import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.storage)
class TorchTensorStorageTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_storage_correctness(self):
    # Define the dimension and size of the tensor
    dim = random.randint(1, 4)
    num_of_elements_each_dim = random.randint(1, 5)
    input_size = [num_of_elements_each_dim for i in range(dim)]

    # Create a random tensor
    tensor = torch.randn(input_size)

    # Get the underlying storage of the tensor
    storage = tensor.storage()

    return storage
