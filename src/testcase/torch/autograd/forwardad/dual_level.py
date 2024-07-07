import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.autograd.forward_ad.dual_level)
class TorchAutogradForwardadDuallevelTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_dual_level_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1,5) # Random number of elements each dimension
        input_size=[num_of_elements_each_dim for i in range(dim)] 
    
        x = torch.randn(input_size, requires_grad=True) # primal input
        x_t = torch.randn(input_size) # tangent
    
        with torch.autograd.forward_ad.dual_level():
            dual_input = torch.autograd.forward_ad.make_dual(x, x_t) # Create a dual input
            output = torch.sum(dual_input) # Do computation with dual_input
            _, grad = torch.autograd.forward_ad.unpack_dual(output) # Unpack primal output and tangent
    
        return grad
            
    
    
    