import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.multinomial)
class TorchTensorMultinomialTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_multinomial_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensor
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for _ in range(dim)]
    
        tensor = torch.randn(input_size).abs()  # Random tensor with non-negative values
        num_samples = random.randint(1, num_of_elements_each_dim)  # Random number of samples to draw
        replacement = random.choice([True, False])  # Randomly choose whether to sample with replacement
    
        result = tensor.multinomial(num_samples, replacement)
        return result
    
    
    
    