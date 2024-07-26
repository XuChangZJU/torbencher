import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.utils.parametrize.ParametrizationList)
class TorchNnUtilsParametrizeParametrizationlistTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_parametrization_list_correctness(self):
        class MyParametrization(nn.Module):
            def forward(self, X):
                return X * 2
    
        class MyModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(3, 3))
    
            def forward(self, X):
                return X @ self.weight
    
        # Create a module and register a parametrization
        module = MyModule()
        parametrize.register_parametrization(module, 'weight', MyParametrization())
    
        # Check the type of the parametrization list
        assert isinstance(module.parametrizations.weight, parametrize.ParametrizationList)
    
        # Check the original parameter is stored correctly
        assert hasattr(module.parametrizations.weight, 'original')
    
        # Generate random input tensor
        input_tensor = torch.randn(3, 3)
    
        # Forward pass through the module
        output = module(input_tensor)
    
        return output
    
    
    
    