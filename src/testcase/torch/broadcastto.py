import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.broadcastto)
class TorchBroadcasttoTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_broadcast_to_correctness(self):
    # Randomly generate dimensions: between 1 and 3
    source_dim = random.randint(1, 3)
    target_dim = random.randint(source_dim, source_dim + 2)
    
    # Randomly generate the original input size
    input_size = [random.randint(1, 3) for _ in range(source_dim)]
    
    # Build the target size based on input_size ensuring valid broadcasting
    # Generating dimensions with 1 to promote broadcasting
    target_size = [1 if i >= len(input_size) else input_size[i] for i in range(target_dim)]

    # Randomly assign a new dimension value where needed
    for i in range(len(target_size)):
        if target_size[i] == 1 and random.uniform(0, 1) > 0.5:
            target_size[i] = random.randint(2, 4)

    input_tensor = torch.randn(input_size)

    # Broadcasting the tensor to the new shape
    result = torch.broadcast_to(input_tensor, target_size)

    return result
