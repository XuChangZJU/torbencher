import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.setdeterministicdebugmode)
class TorchSetdeterministicdebugmodeTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_set_deterministic_debug_mode(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for _ in range(dim)]  # Generate random input size
    
        tensor1 = torch.randn(input_size)
        tensor2 = torch.randn(input_size)
        alpha = random.uniform(0.1, 10.0)  # Random alpha value between 0.1 and 10.0
    
        # Test all possible modes
        for mode in ["default", "warn", "error"]:
            torch.set_deterministic_debug_mode(mode)
            try:
                result = torch.add(tensor1, tensor2, alpha=alpha)
            except RuntimeError as e:
                result = str(e)  # Capture the error as the result
    
        return result
    