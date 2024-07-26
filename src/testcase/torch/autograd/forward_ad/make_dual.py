import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.autograd.forward_ad.make_dual)
class TorchAutogradForwardadMakedualTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_make_dual_correctness(self):
        # Random dimension for the tensors
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]
    
        # Random tensor for the primary value
        tensor = torch.randn(input_size)
        # Random tensor for the tangent value, must be the same size as tensor
        tangent = torch.randn(input_size)
    
        # Enter dual level
        with torch.autograd.forward_ad.dual_level():
            # Create dual tensor
            dual_tensor = torch.autograd.forward_ad.make_dual(tensor, tangent)
        
        return dual_tensor
    