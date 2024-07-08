import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.autograd.graph.saved_tensors_hooks)
class TorchAutogradGraphSavedtensorshooksTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_saved_tensors_hooks_correctness(self):
        # Define pack and unpack hooks
        def pack_hook(tensor):
            "Packing", tensor
            return tensor
    
        def unpack_hook(packed_tensor):
            "Unpacking", packed_tensor
            return packed_tensor
        
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1,5) # Random number of elements each dimension
        input_size=[num_of_elements_each_dim for i in range(dim)] 
    
        a = torch.randn(input_size, requires_grad=True)
        b = torch.randn(input_size, requires_grad=True)
        with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
            y = a * b
        result = y.sum().backward()
        return result
    