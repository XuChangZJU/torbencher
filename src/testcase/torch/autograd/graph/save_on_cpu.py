import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.autograd.graph.save_on_cpu)
class TorchAutogradGraphSaveoncpuTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_save_on_cpu_correctness(self):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Please ensure that your PyTorch installation is compiled with CUDA support.")
    
        # Random dimension for the tensors
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]
    
        # Random tensors with requires_grad=True and on CUDA device
        tensor_a = torch.randn(input_size, requires_grad=True, device="cuda")
        tensor_b = torch.randn(input_size, requires_grad=True, device="cuda")
        tensor_c = torch.randn(input_size, requires_grad=True, device="cuda")
    
        def f(a, b, c):
            prod_1 = a * b  # a and b are saved on GPU
            with torch.autograd.graph.save_on_cpu():
                prod_2 = prod_1 * c  # prod_1 and c are saved on CPU
            y = prod_2 * a  # prod_2 and a are saved on GPU
            return y
    
        y = f(tensor_a, tensor_b, tensor_c)
        del tensor_a, tensor_b, tensor_c  # for illustration only
        y.sum().backward()  # all CPU tensors are moved back to GPU, for backward
    