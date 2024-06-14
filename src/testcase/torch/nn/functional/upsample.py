import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.upsample)
class TorchNnFunctionalUpsampleTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_upsample_correctness(self):
    # Random input size
    dim = random.randint(3, 5)  # 3D, 4D or 5D input
    num_of_elements_each_dim = random.randint(1, 5)
    input_size = [num_of_elements_each_dim for i in range(dim)]

    # Random input tensor
    input_tensor = torch.randn(input_size)

    # Randomly choose between 'size' and 'scale_factor'
    choice = random.choice(['size', 'scale_factor'])

    if choice == 'size':
        # Random output size
        output_size = [random.randint(1, 10) for _ in range(dim - 2)]  # Spatial dimensions only
        result = torch.nn.functional.upsample(input_tensor, size=output_size)
    else:
        # Random scale factor
        scale_factor = random.uniform(1.0, 3.0)  # Upsampling
        result = torch.nn.functional.upsample(input_tensor, scale_factor=scale_factor)
    
    return result
