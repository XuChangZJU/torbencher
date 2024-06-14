import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.utils.prune.lnstructured)
class TorchNnUtilsPruneLnstructuredTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_ln_structured_correctness(self):
    # Randomly generate dimensions for the Conv2d layer
    in_channels = random.randint(1, 10)
    out_channels = random.randint(1, 10)
    kernel_size = random.randint(1, 5)
    
    # Create a Conv2d module with random dimensions
    module = nn.Conv2d(in_channels, out_channels, kernel_size)
    
    # Randomly select the parameter name to prune
    param_name = 'weight'
    
    # Randomly generate the amount to prune (as a fraction)
    amount = random.uniform(0.1, 0.5)
    
    # Randomly select the norm type
    norm_type = random.choice([1, 2, float('inf'), float('-inf'), 'fro', 'nuc'])
    
    # Randomly select the dimension along which to prune
    dim = random.randint(0, 1)
    
    # Apply ln_structured pruning
    pruned_module = prune.ln_structured(module, param_name, amount, norm_type, dim)
    
    # Return the pruned module to observe the effect
    return pruned_module
