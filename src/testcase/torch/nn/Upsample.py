import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.Upsample)
class TorchNnUpsampleTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_upsample_correctness(self):
    # Randomly choose the dimension of the input tensor (1D, 2D, or 3D)
    dim = random.choice([3, 4, 5])
    
    # Randomly choose the size of each dimension
    input_size = [random.randint(1, 5) for _ in range(dim)]
    
    # Create a random tensor with the chosen size
    input_tensor = torch.randn(input_size)
    
    # Randomly choose a scale factor between 1.1 and 3.0
    scale_factor = random.uniform(1.1, 3.0)
    
    # Randomly choose an upsampling mode
    mode = random.choice(['nearest', 'linear', 'bilinear', 'bicubic', 'trilinear'])
    
    # Create the Upsample module with the chosen parameters
    upsample = torch.nn.Upsample(scale_factor=scale_factor, mode=mode)
    
    # Apply the upsample operation to the input tensor
    result = upsample(input_tensor)
    
    return result
