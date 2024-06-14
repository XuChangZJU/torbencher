import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.cpu)
class TorchTensorCpuTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_cpu_correctness(self):
        # Randomly generate tensor dimension and size
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
    
        # Create a random tensor
        input_tensor = torch.randn(input_size)
    
        # Move tensor to cpu
        result = input_tensor.cpu()
        return result
    