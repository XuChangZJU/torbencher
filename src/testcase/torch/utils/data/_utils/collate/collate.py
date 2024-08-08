import random

import torch
from torch.utils.data._utils.collate import collate

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.utils.data._utils.collate.collate)
class TorchUtilsDataUutilsCollateCollateTestCase(TorBencherTestCaseBase):
    @staticmethod
    @test_api_version.larger_than("2.0.0")
    def collate_tensor_fn(batch, *, collate_fn_map):
        return torch.stack(batch, 0)

    def test_collate_correctness(self):
        # Randomly generate the number of tensors in the batch
        batch_size = random.randint(2, 5)

        # Randomly generate the dimension of each tensor
        dim = random.randint(1, 4)

        # Randomly generate the size of each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        tensor_size = [num_of_elements_each_dim for _ in range(dim)]

        # Create a batch of random tensors
        batch = [torch.randn(tensor_size) for _ in range(batch_size)]

        # Define the custom collate function map
        collate_fn_map = {torch.Tensor: self.collate_tensor_fn}

        # Call the collate function
        result = collate(batch, collate_fn_map=collate_fn_map)

        return result
