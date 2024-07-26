import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.autograd.forward_ad.enter_dual_level)
class TorchAutogradForwardadEnterduallevelTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_enter_dual_level_correctness(self):
        # Enter a new forward grad level
        with torch.autograd.forward_ad.dual_level() as level:
            # Random dimension for the tensor
            dim = random.randint(1, 4)
            # Random number of elements each dimension
            num_of_elements_each_dim = random.randint(1, 5)
            input_size = [num_of_elements_each_dim for _ in range(dim)]
            
            # Create a random tensor
            tensor = torch.randn(input_size)
            
            # Make a dual tensor
            dual_tensor = torch.autograd.forward_ad.make_dual(tensor, torch.randn(input_size))
            
            # Unpack the dual tensor
            primal, tangent = torch.autograd.forward_ad.unpack_dual(dual_tensor)
            
            # Check if the primal and tangent parts are correctly unpacked
            assert torch.allclose(primal, tensor), "Primal part does not match the original tensor"
            assert tangent.shape == tensor.shape, "Tangent part shape does not match the original tensor shape"
            
            return primal, tangent
    