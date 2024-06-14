import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.FractionalMaxPool3d)
class TorchNnFractionalmaxpool3dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_fractional_max_pool3d_correctness(self):
        # Randomly generate dimensions for the input tensor
        batch_size = random.randint(1, 5)  # Random batch size
        channels = random.randint(1, 5)  # Random number of channels
        depth = random.randint(10, 20)  # Random depth of the input tensor
        height = random.randint(10, 20)  # Random height of the input tensor
        width = random.randint(10, 20)  # Random width of the input tensor
    
        # Generate random input tensor with the specified dimensions
        input_tensor = torch.randn(batch_size, channels, depth, height, width)
    
        # Randomly generate kernel size for the pooling operation
        kernel_size = random.randint(2, 5)  # Random kernel size
    
        # Randomly decide whether to use output_size or output_ratio
        if random.choice([True, False]):
            # Use output_size
            output_depth = random.randint(5, depth - 1)  # Ensure output depth is less than input depth
            output_height = random.randint(5, height - 1)  # Ensure output height is less than input height
            output_width = random.randint(5, width - 1)  # Ensure output width is less than input width
            pool_layer = torch.nn.FractionalMaxPool3d(kernel_size, (output_depth, output_height, output_width))
        else:
            # Use output_ratio
            output_ratio_depth = random.uniform(0.1, 0.9)  # Random ratio for depth
            output_ratio_height = random.uniform(0.1, 0.9)  # Random ratio for height
            output_ratio_width = random.uniform(0.1, 0.9)  # Random ratio for width
            pool_layer = torch.nn.FractionalMaxPool3d(kernel_size, output_ratio=(output_ratio_depth, output_ratio_height, output_ratio_width))
    
        # Apply the fractional max pooling operation
        output_tensor = pool_layer(input_tensor)
        return output_tensor
    
    
    
    