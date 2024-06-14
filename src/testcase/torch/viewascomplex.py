import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.viewascomplex)
class TorchViewascomplexTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_view_as_complex_correctness(self):
        # Random dimension for the tensors (at least 1)
        dim = random.randint(1, 5)
        # Random number of elements each dimension (at least 1)
        num_of_elements_each_dim = random.randint(1, 5)
        # Generate input size list
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Append 2 to the end of the list to represent real and imaginary components
        input_size.append(2)
        # Generate random float tensor with the specified input size
        input_tensor = torch.randn(input_size)
        # Call view_as_complex function
        result = torch.view_as_complex(input_tensor)
        return result
    