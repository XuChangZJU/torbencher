import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.utils.prune.l1_unstructured)
class TorchNnUtilsPruneL1unstructuredTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_l1_unstructured_correctness(self):
        # Randomly generate dimensions for the Linear layer
        in_features = random.randint(1, 10)
        out_features = random.randint(1, 10)
        
        # Create a random Linear module
        module = nn.Linear(in_features, out_features)
        
        # Randomly choose the amount to prune
        amount = random.uniform(0.1, 0.5)  # Fraction of parameters to prune
        
        # Apply L1 unstructured pruning
        pruned_module = prune.l1_unstructured(module, 'weight', amount)
        
        # Check the state_dict to ensure the mask and original weights are present
        state_dict_keys = pruned_module.state_dict().keys()
        
        return state_dict_keys
    
    
    
    