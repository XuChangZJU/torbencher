import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.AdaptiveAvgPool3d)
class TorchNnAdaptiveavgpool3dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_AdaptiveAvgPool3d_correctness(self):
        # Random input size
        dim = 5
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
    
        # Random input tensor
        input_tensor = torch.randn(input_size)
    
        # Random output size
        output_size = [random.randint(1, input_size[i]) for i in range(2, 5)]
    
        # AdaptiveAvgPool3d operator
        adaptive_avg_pool_3d = torch.nn.AdaptiveAvgPool3d(output_size)
    
        # Output tensor
        output_tensor = adaptive_avg_pool_3d(input_tensor)
        return output_tensor
    
    
    
    