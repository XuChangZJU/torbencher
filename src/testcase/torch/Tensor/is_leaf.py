import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.is_leaf)
class TorchTensorIsleafTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_is_leaf_correctness(self):
        # Check if CUDA is available
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Please ensure that your PyTorch installation is compiled with CUDA support.")
    
        # Random dimension for the tensors
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]
    
        # Case 1: Tensor with requires_grad=True created by the user
        tensor_a = torch.randn(input_size, requires_grad=True)
        result_a = tensor_a.is_leaf
    
        # Case 2: Tensor with requires_grad=True moved to CUDA
        tensor_b = torch.randn(input_size, requires_grad=True).cuda()
        result_b = tensor_b.is_leaf
    
        # Case 3: Tensor created by an operation (addition)
        tensor_c = torch.randn(input_size, requires_grad=True) + 2
        result_c = tensor_c.is_leaf
    
        # Case 4: Tensor without requires_grad moved to CUDA
        tensor_d = torch.randn(input_size).cuda()
        result_d = tensor_d.is_leaf
    
        # Case 5: Tensor without requires_grad, then requires_grad set to True
        tensor_e = torch.randn(input_size).cuda().requires_grad_()
        result_e = tensor_e.is_leaf
    
        # Case 6: Tensor with requires_grad=True created directly on CUDA
        tensor_f = torch.randn(input_size, requires_grad=True, device="cuda")
        result_f = tensor_f.is_leaf
    
        return result_a, result_b, result_c, result_d, result_e, result_f
    
    
    
    