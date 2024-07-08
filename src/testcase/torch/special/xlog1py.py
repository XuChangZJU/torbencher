import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.special.xlog1py)
class TorchSpecialXlog1pyTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_xlog1py_correctness(self):
        # Random dimension for the tensors
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]
    
        # Generate random tensors for input and other
        input_tensor = torch.randn(input_size)
        other_tensor = torch.randn(input_size)
    
        # Compute the result using torch.special.xlog1py
        result = torch.special.xlog1py(input_tensor, other_tensor)
        return result
    def test_xlog1py_edge_cases(self):
        # Test case with zeros in input tensor
        input_tensor = torch.zeros(5)
        other_tensor = torch.tensor([-1, 0, 1, float('inf'), float('nan')])
        result_zeros = torch.special.xlog1py(input_tensor, other_tensor)
    
        # Test case with positive integers in input tensor
        input_tensor = torch.tensor([1, 2, 3])
        other_tensor = torch.tensor([3, 2, 1])
        result_positive_integers = torch.special.xlog1py(input_tensor, other_tensor)
    
        # Test case with scalar other
        input_tensor = torch.tensor([1, 2, 3])
        other_scalar = 4
        result_scalar_other = torch.special.xlog1py(input_tensor, other_scalar)
    
        # Test case with scalar input
        input_scalar = 2
        other_tensor = torch.tensor([3, 2, 1])
        result_scalar_input = torch.special.xlog1py(input_scalar, other_tensor)
    
        return result_zeros, result_positive_integers, result_scalar_other, result_scalar_input
    