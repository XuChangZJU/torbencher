import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.utils.prune.custom_from_mask)
class TorchNnUtilsPruneCustomUfromUmaskTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_custom_from_mask_correctness(self):
        # Random dimension for the weight tensor
        dim = random.randint(1, 4)
        # Random number of elements each dimension for the weight tensor
        num_of_elements_each_dim = random.randint(1, 5)
        # Random input size for the weight tensor
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Randomly generated weight tensor
        weight_tensor = torch.randn(input_size)
        # Create a Linear module
        linear_module = torch.nn.Linear(in_features=num_of_elements_each_dim, out_features=num_of_elements_each_dim)
        # Assign the random weight tensor to the Linear module
        linear_module.weight = torch.nn.Parameter(weight_tensor)
        # Randomly generate a binary mask for pruning
        mask = torch.randint(0, 2, weight_tensor.size(), dtype=torch.bool)
        # Apply pruning using the custom_from_mask function
        pruned_module = torch.nn.utils.prune.custom_from_mask(linear_module, 'weight', mask)
        # Return the pruned module, pruned weight, and the mask
        return pruned_module, pruned_module.weight, pruned_module.weight_mask
