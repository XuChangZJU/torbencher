import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Generator)
class TorchGeneratorTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_generator_correctness(self):
        # Randomly choose between 'cpu' and 'cuda' device
        device_type = 'cpu' if random.random() < 0.5 else 'cuda'
        
        # Create generator on the chosen device
        generator = torch.Generator(device_type)
    
        # Use the generator to create a seeded random tensor
        seed = random.randint(1, 1000)
        generator.manual_seed(seed)
        
        tensor_size = [random.randint(1, 5) for _ in range(random.randint(1, 4))]
        tensor_with_generator = torch.randn(tensor_size, generator=generator)
        
        # Check the reproducibility with the same seed
        generator.manual_seed(seed)
        tensor_with_same_generator = torch.randn(tensor_size, generator=generator)
    
        # Verify if the tensors from the same seed are equal
        return torch.equal(tensor_with_generator, tensor_with_same_generator)
    