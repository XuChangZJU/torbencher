import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.expandas)
class TorchTensorExpandasTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_expand_as_correctness(self):
        # Generate random dimensions for the base tensor
        base_dim = random.randint(1, 4)
        base_size = [random.randint(1, 5) for _ in range(base_dim)]
        
        # Generate random dimensions for the target tensor
        target_dim = random.randint(base_dim, base_dim + 2)  # Ensure target_dim >= base_dim
        target_size = base_size + [random.randint(1, 5) for _ in range(target_dim - base_dim)]
        
        # Create base and target tensors
        base_tensor = torch.randn(base_size)
        target_tensor = torch.randn(target_size)
        
        # Expand base_tensor to the size of target_tensor
        result = base_tensor.expand_as(target_tensor)
        return result
    