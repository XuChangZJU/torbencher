import torch
import random
from torch.autograd.forward_ad import unpack_dual

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.autograd.forward_ad.UnpackedDualTensor)
class TorchAutogradForwardadUnpackeddualtensorTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_unpack_dual_correctness(self):
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
    
        primal = torch.randn(input_size)
        tangent = torch.randn(input_size)
        dual_tensor = torch.autograd.forward_ad.make_dual(primal, tangent)
        unpacked_dual_tensor = unpack_dual(dual_tensor)
        return unpacked_dual_tensor
    