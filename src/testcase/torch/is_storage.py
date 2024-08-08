import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.is_storage)
class TorchIsUstorageTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_is_storage_correctness(self):
        # Create a random tensor
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
        tensor = torch.randn(input_size)

        # Get the storage of the tensor
        storage = tensor.storage()

        # Test if the tensor is a storage object (should be False)
        result1 = torch.is_storage(tensor)

        # Test if the storage is a storage object (should be True)
        result2 = torch.is_storage(storage)

        return result1, result2
