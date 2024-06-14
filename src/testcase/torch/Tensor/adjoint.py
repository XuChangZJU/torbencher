import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.adjoint)
class TorchTensorAdjointTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_adjoint_correctness(self):
        # Random dimension for the tensors
        dim = random.randint(2, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Generate random size for the tensor
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Generate a random tensor with complex data type
        input_tensor = torch.randn(input_size, dtype=torch.cdouble)
        # Calculate the adjoint of the tensor
        result = input_tensor.adjoint()
        return result
    
    
    
    