import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.broadcastto)
class TorchTensorBroadcasttoTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_broadcast_to_correctness(self):
    # Randomly generate the dimensions for the original tensor
    original_dim = random.randint(1, 4)
    original_size = [random.randint(1, 5) for _ in range(original_dim)]
    
    # Create the original tensor with random values
    original_tensor = torch.randn(original_size)
    
    # Generate a valid target shape for broadcasting
    target_dim = random.randint(original_dim, original_dim + 2)  # Ensure target_dim >= original_dim
    target_shape = [1] * (target_dim - original_dim) + original_size  # Start with original size
    for i in range(target_dim):
        if random.random() > 0.5:  # Randomly decide to expand dimensions
            target_shape[i] = random.randint(max(1, target_shape[i]), 10)  # Ensure valid broadcasting
    
    # Perform the broadcast operation
    result = original_tensor.broadcast_to(target_shape)
    return result
