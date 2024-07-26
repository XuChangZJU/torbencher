import torch
import random
import torch.nn.utils.prune as prune

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.utils.prune.L1Unstructured)
class TorchNnUtilsPruneL1unstructuredTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_L1Unstructured_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensor
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for _ in range(dim)]

        tensor = torch.randn(input_size)
        amount = random.uniform(0.0, 1.0) if random.choice([True, False]) else random.randint(1,
                                                                                              tensor.numel())  # Random amount to prune

        pruner = prune.L1Unstructured(amount)
        pruned_tensor = pruner(tensor)

        return pruned_tensor
