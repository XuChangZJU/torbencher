import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.cuda)
class TorchTensorCudaTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_tensor_cuda_correctness(self):
        # Random dimension for the tensor
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]
    
        # Generate a random tensor
        tensor = torch.randn(input_size)
    
        # Randomly select a CUDA device if available
        if torch.cuda.is_available():
            device_id = random.randint(0, torch.cuda.device_count() - 1)
            device = torch.device(f'cuda:{device_id}')
        else:
            device = torch.device('cpu')
    
        # Move tensor to the selected device
        result = tensor.to(device)
        return result
    
    
    
    