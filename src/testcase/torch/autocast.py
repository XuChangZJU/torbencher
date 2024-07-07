import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.autocast)
class TorchAutocastTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_autocast_correctness_cuda(self):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Please ensure that your PyTorch installation is compiled with CUDA support.")
    
        # Randomly generate input size for tensors
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]
    
        # Create random tensors
        tensor1 = torch.randn(input_size, device="cuda")
        tensor2 = torch.randn(input_size, device="cuda")
    
        # Create a simple model
        model = torch.nn.Linear(num_of_elements_each_dim, num_of_elements_each_dim).cuda()
    
        # Enable autocasting for the forward pass
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = model(tensor1)
            result = torch.add(output, tensor2)
    
        return result
    
    def test_autocast_correctness_cpu(self):
        # Randomly generate input size for tensors
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]
    
        # Create random tensors
        tensor1 = torch.randn(input_size)
        tensor2 = torch.randn(input_size)
    
        # Create a simple model
        model = torch.nn.Linear(num_of_elements_each_dim, num_of_elements_each_dim)
    
        # Enable autocasting for the forward pass
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            output = model(tensor1)
            result = torch.add(output, tensor2)
    
        return result
    
    
    