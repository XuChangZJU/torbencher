import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.utils.prune.CustomFromMask)
class TorchNnUtilsPruneCustomfrommaskTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_custom_from_mask_correctness(self):
    # Randomly generate dimensions for the tensor
    dim = random.randint(1, 4)
    num_of_elements_each_dim = random.randint(1, 5)
    input_size = [num_of_elements_each_dim for _ in range(dim)]

    # Randomly generate a tensor and a mask of the same size
    tensor = torch.randn(input_size)
    mask = torch.randint(0, 2, input_size).float()  # Mask should be binary (0 or 1)

    # Apply CustomFromMask pruning
    prune = torch.nn.utils.prune.CustomFromMask(tensor, mask)
    result = prune.apply(tensor, mask)
    
    return result
