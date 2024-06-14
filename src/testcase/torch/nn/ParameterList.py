import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.ParameterList)
class TorchNnParameterlistTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_parameterlist_correctness(self):
        # Random number of parameters in the ParameterList
        num_params = random.randint(1, 5)
        
        # Random dimensions for each parameter tensor
        param_dim1 = random.randint(1, 4)
        param_dim2 = random.randint(1, 4)
        
        # Create a ParameterList with random tensors
        parameters = [torch.nn.Parameter(torch.randn(param_dim1, param_dim2)) for _ in range(num_params)]
        param_list = torch.nn.ParameterList(parameters)
        
        # Create a random input tensor with compatible dimensions
        input_tensor = torch.randn(param_dim2, random.randint(1, 4))
        
        # Perform a simple operation using the ParameterList
        result = input_tensor
        for param in param_list:
            result = param.mm(result)
        
        return result
    