import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.init.calculate_gain)
class TorchNnInitCalculategainTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_calculate_gain_correctness(self):
        # Define possible nonlinearities and their expected gains
        nonlinearities = {
            'linear': 1,
            'conv1d': 1,
            'conv2d': 1,
            'conv3d': 1,
            'sigmoid': 1,
            'tanh': 5/3,
            'relu': torch.sqrt(torch.tensor(2.)),
            'leaky_relu': torch.sqrt(torch.tensor(2. / (1 + 0.2**2))),  # Example negative_slope = 0.2
            'selu': 3/4
        }
    
        # Randomly choose a nonlinearity
        nonlinearity = random.choice(list(nonlinearities.keys()))
    
        # Calculate gain using the function
        if nonlinearity == 'leaky_relu':
            negative_slope = random.uniform(0.01, 1.0)  # Ensure negative slope is not zero
            gain = torch.nn.init.calculate_gain(nonlinearity, negative_slope)
        else:
            gain = torch.nn.init.calculate_gain(nonlinearity)
    
        return gain
    
    # Automatically added function calls
    
    
    