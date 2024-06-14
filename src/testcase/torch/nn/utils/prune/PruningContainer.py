import torch
import random
import torch.nn.utils.prune as prune


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.utils.prune.PruningContainer)
class TorchNnUtilsPrunePruningcontainerTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_pruning_container_correctness(self):
    # Randomly generate tensor dimensions and size
    dim = random.randint(1, 4)
    num_of_elements_each_dim = random.randint(1, 5)
    input_size = [num_of_elements_each_dim for _ in range(dim)]

    # Create a random tensor
    tensor = torch.randn(input_size)

    # Create a random mask with the same size as the tensor
    mask = torch.randint(0, 2, input_size).float()

    # Create a pruning method (using L1Unstructured as an example)
    pruning_method = prune.L1Unstructured(amount=random.uniform(0.1, 0.5))

    # Create a PruningContainer and add the pruning method
    pruning_container = prune.PruningContainer([pruning_method])

    # Apply the pruning method to the tensor
    pruned_tensor = pruning_container.apply(tensor, mask)

    return pruned_tensor
