import random

import torch
import torch.utils.data

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.utils.data.Subset)
class TorchUtilsDataSubsetTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_subset_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]
        dataset_tensor = torch.randn(input_size)
        dataset = torch.utils.data.TensorDataset(dataset_tensor)
        # Generate random indices for subset, ensuring they are within the dataset bounds, do not use random.sample
        indices = [random.randint(0, len(dataset) - 1) for _ in range(random.randint(1, len(dataset)))]
        subset = torch.utils.data.Subset(dataset, indices)
        # extract tensors from the subset
        subset_tensors = [dataset[i] for i in subset.indices]
        return subset_tensors
        