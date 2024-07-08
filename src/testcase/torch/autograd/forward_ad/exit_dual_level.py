import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.autograd.forward_ad.exit_dual_level)
class TorchAutogradForwardadExitduallevelTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_exit_dual_level_correctness(self):
        # Create a random tensor
        dim = random.randint(1, 4)  # Random dimension for the tensor
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for _ in range(dim)]
        tensor = torch.randn(input_size, requires_grad=True)
    
        # Enter a forward grad level
        with torch.autograd.forward_ad.dual_level(1):  # Ensure the level is a positive number
            # Perform some operations within the forward grad level
            tensor_squared = tensor * tensor
    
        # Perform another operation to check if the level has been exited
        tensor_cubed = tensor * tensor * tensor
    
        return tensor_squared, tensor_cubed
    