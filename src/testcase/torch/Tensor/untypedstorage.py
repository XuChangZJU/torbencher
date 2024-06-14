import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.untypedstorage)
class TorchTensorUntypedstorageTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_untyped_storage_correctness(self):
    dim = random.randint(1, 4)  # Random dimension for the tensor
    num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
    input_size = [num_of_elements_each_dim for i in range(dim)]
    tensor = torch.randn(input_size)  # Create a random tensor
    untyped_storage = tensor.untyped_storage()  # Get the underlying UntypedStorage
    return untyped_storage
