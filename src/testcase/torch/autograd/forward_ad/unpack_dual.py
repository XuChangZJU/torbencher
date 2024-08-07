import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.autograd.forward_ad.unpack_dual)
class TorchAutogradForwardUadUnpackUdualTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_unpack_dual_correctness(self):
        # Random dimension for the tensors
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]

        # Random primal tensor
        primal_tensor = torch.randn(input_size)
        # Random tangent tensor
        tangent_tensor = torch.randn(input_size)

        # Create a dual tensor using make_dual
        with torch.autograd.forward_ad.dual_level():
            dual_tensor = torch.autograd.forward_ad.make_dual(primal_tensor, tangent_tensor)
            # Unpack the dual tensor
            primal, tangent = torch.autograd.forward_ad.unpack_dual(dual_tensor)

        return primal, tangent
